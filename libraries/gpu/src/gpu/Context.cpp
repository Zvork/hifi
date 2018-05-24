//
//  Context.cpp
//  interface/src/gpu
//
//  Created by Sam Gateau on 10/27/2014.
//  Copyright 2014 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#include "Context.h"

#include <shared/GlobalAppProperties.h>

#include "Frame.h"
#include "GPULogging.h"

using namespace gpu;

void ContextStats::evalDelta(const ContextStats& begin, const ContextStats& end) {
    _ISNumFormatChanges = end._ISNumFormatChanges - begin._ISNumFormatChanges;
    _ISNumInputBufferChanges = end._ISNumInputBufferChanges - begin._ISNumInputBufferChanges;
    _ISNumIndexBufferChanges = end._ISNumIndexBufferChanges - begin._ISNumIndexBufferChanges;

    _RSNumTextureBounded = end._RSNumTextureBounded - begin._RSNumTextureBounded;
    _RSAmountTextureMemoryBounded = end._RSAmountTextureMemoryBounded - begin._RSAmountTextureMemoryBounded;

    _DSNumAPIDrawcalls = end._DSNumAPIDrawcalls - begin._DSNumAPIDrawcalls;
    _DSNumDrawcalls = end._DSNumDrawcalls - begin._DSNumDrawcalls;
    _DSNumTriangles= end._DSNumTriangles - begin._DSNumTriangles;

    _PSNumSetPipelines = end._PSNumSetPipelines - begin._PSNumSetPipelines;
}


Context::CreateBackend Context::_createBackendCallback = nullptr;
Context::MakeProgram Context::_makeProgramCallback = nullptr;
std::once_flag Context::_initialized;

Context::Context() {
    if (_createBackendCallback) {
        _backend = _createBackendCallback();
    }
}

Context::Context(const Context& context) {
}

Context::~Context() {
    for (auto batch : _batchPool) {
        delete batch;
    }
    _batchPool.clear();
}

const std::string& Context::getBackendVersion() const {
    return _backend->getVersion();
}

void Context::beginFrame(const glm::mat4& renderView, const glm::mat4& renderPose) {
    assert(!_frameActive);
    _frameActive = true;
    _currentFrame = std::make_shared<Frame>();
    _currentFrame->pose = renderPose;
    _currentFrame->view = renderView;

    if (!_frameRangeTimer) {
        _frameRangeTimer = std::make_shared<RangeTimer>("gpu::Context::Frame");
    }
}

void Context::appendFrameBatch(const BatchPointer& batch) {
    if (!_frameActive) {
        qWarning() << "Batch executed outside of frame boundaries";
        return;
    }
    _currentFrame->batches.push_back(batch);
}

FramePointer Context::endFrame() {
    assert(_frameActive);
    auto result = _currentFrame;
    _currentFrame.reset();
    _frameActive = false;

    result->stereoState = _stereo;
    result->finish();
    return result;
}

void Context::executeBatch(Batch& batch) const {
    PROFILE_RANGE(render_gpu, __FUNCTION__);
    batch.flush();
    _backend->render(batch);
}

void Context::recycle() const {
    _backend->recycle();
}

void Context::consumeFrameUpdates(const FramePointer& frame) const {
    frame->preRender();
}

void Context::executeFrame(const FramePointer& frame) const {
    PROFILE_RANGE(render_gpu, __FUNCTION__);

    // Grab the stats at the around the frame and delta to have a consistent sampling
    ContextStats beginStats;
    getStats(beginStats);

    // FIXME? probably not necessary, but safe
    consumeFrameUpdates(frame);
    _backend->setStereoState(frame->stereoState);
    {
        Batch beginBatch("Context::executeFrame::begin");
        _frameRangeTimer->begin(beginBatch);
        _backend->render(beginBatch);

        // Execute the frame rendering commands
        for (auto& batch : frame->batches) {
            _backend->render(*batch);
        }

        Batch endBatch("Context::executeFrame::end");
        _frameRangeTimer->end(endBatch);
        _backend->render(endBatch);
    }

    ContextStats endStats;
    getStats(endStats);
    _frameStats.evalDelta(beginStats, endStats);
}

bool Context::makeProgram(Shader& shader, const Shader::BindingSet& bindings, const Shader::CompilationHandler& handler) {
    PROFILE_RANGE_EX(app, "makeProgram", 0xff4040c0, shader.getID());
    // If we're running in another DLL context, we need to fetch the program callback out of the application
    // FIXME find a way to do this without reliance on Qt app properties
    if (!_makeProgramCallback) {
        void* rawCallback = qApp->property(hifi::properties::gl::MAKE_PROGRAM_CALLBACK).value<void*>();
        _makeProgramCallback = reinterpret_cast<Context::MakeProgram>(rawCallback);
    }
    if (shader.isProgram() && _makeProgramCallback) {
        return _makeProgramCallback(shader, bindings, handler);
    }
    return false;
}

void Context::enableStereo(bool enable) {
    _stereo._enable = enable;
}

bool Context::isStereo() {
    return _stereo.isStereo();
}

void Context::setStereoProjections(const mat4 eyeProjections[2]) {
    for (int i = 0; i < 2; ++i) {
        _stereo._eyeProjections[i] = eyeProjections[i];
    }
}

void Context::setStereoViews(const mat4 views[2]) {
    for (int i = 0; i < 2; ++i) {
        _stereo._eyeViews[i] = views[i];
    }
}

void Context::getStereoProjections(mat4* eyeProjections) const {
    for (int i = 0; i < 2; ++i) {
        eyeProjections[i] = _stereo._eyeProjections[i];
    }
}

void Context::getStereoViews(mat4* eyeViews) const {
    for (int i = 0; i < 2; ++i) {
        eyeViews[i] = _stereo._eyeViews[i];
    }
}

void Context::downloadFramebuffer(const FramebufferPointer& srcFramebuffer, const Vec4i& region, QImage& destImage) {
    _backend->downloadFramebuffer(srcFramebuffer, region, destImage);
}

void Context::resetStats() const {
    _backend->resetStats();
}

void Context::getStats(ContextStats& stats) const {
    _backend->getStats(stats);
}

void Context::getFrameStats(ContextStats& stats) const {
    stats = _frameStats;
}

double Context::getFrameTimerGPUAverage() const {
    if (_frameRangeTimer) {
        return _frameRangeTimer->getGPUAverage();
    }
    return 0.0;
}

double Context::getFrameTimerBatchAverage() const {
    if (_frameRangeTimer) {
        return _frameRangeTimer->getBatchAverage();
    }
    return 0.0;
}

const Backend::TransformCamera& Backend::TransformCamera::recomputeDerived(const Transform& view, const Transform& previousView) const {
    _projectionInverse = glm::inverse(_projection);

    // Get the viewEyeToWorld matrix form the transformView as passed to the gpu::Batch
    // this is the "_viewInverse" fed to the shader
    // Genetrate the "_view" matrix as well from the xform
    view.getMatrix(_viewInverse);
    _view = glm::inverse(_viewInverse);
    previousView.getMatrix(_previousViewInverse);
    _previousView = glm::inverse(_previousViewInverse);

    Mat4 viewUntranslated = _view;
    viewUntranslated[3] = Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    _projectionViewUntranslated = _projection * viewUntranslated;

    viewUntranslated = _previousView;
    viewUntranslated[3] = Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    _previousProjectionViewUntranslated = _previousProjection * viewUntranslated;

    _stereoInfo = Vec4(0.0f);

    return *this;
}

Backend::TransformCamera Backend::TransformCamera::getEyeCamera(int eye, const StereoState& _stereo, 
                                                                const Transform& view, const Transform& previousView, 
                                                                Vec2 normalizedJitter, Vec2 normalizedPrevJitter) const {
    TransformCamera result = *this;
    Transform eyeView = view;
    Transform eyePreviousView = previousView;
    if (!_stereo._skybox) {
        eyeView.postTranslate(-Vec3(_stereo._eyeViews[eye][3]));
        eyePreviousView.postTranslate(-Vec3(_stereo._eyeViews[eye][3]));
    } else {
        // FIXME: If "skybox" the ipd is set to 0 for now, let s try to propose a better solution for this in the future
    }
    result._projection = _stereo._eyeProjections[eye];
    // We suppose that the projection, except for jitter, hasn't changed from previous frame
    result._previousProjection = result._projection;

    result._jitterUV = Vec4(normalizedJitter, normalizedPrevJitter);

    // Apply jitter to projections
    normalizedJitter.x *= 2.0f;
    normalizedPrevJitter.x *= 2.0f;
    result._projection[2][0] += normalizedJitter.x;
    result._projection[2][1] += normalizedJitter.y;
    result._previousProjection[2][0] += normalizedPrevJitter.x;
    result._previousProjection[2][1] += normalizedPrevJitter.y;

    result.recomputeDerived(eyeView, eyePreviousView);

    result._stereoInfo = Vec4(1.0f, (float)eye, 1.0f / result._viewport.z, 1.0f / result._viewport.w);

    return result;
}

Backend::TransformCamera Backend::TransformCamera::getMonoCamera(const Transform& view, const Transform& previousView, 
                                                                 Vec2 normalizedJitter, Vec2 normalizedPrevJitter) const {
    TransformCamera result = *this;
    // We suppose that the projection, except for jitter, hasn't changed from previous frame
    result._previousProjection = result._projection;
    result._projection[2][0] += normalizedJitter.x;
    result._projection[2][1] += normalizedJitter.y;
    result._previousProjection[2][0] += normalizedPrevJitter.x;
    result._previousProjection[2][1] += normalizedPrevJitter.y;
    result.recomputeDerived(view, previousView);

    result._stereoInfo = Vec4(0.0f, 0.0f, 1.0f / result._viewport.z, 1.0f / result._viewport.w);
    result._jitterUV = Vec4(normalizedJitter, normalizedPrevJitter);
    return result;
}

// Counters for Buffer and Texture usage in GPU/Context

ContextMetricSize  Backend::freeGPUMemSize;

ContextMetricCount Backend::bufferCount;
ContextMetricSize  Backend::bufferGPUMemSize;

ContextMetricCount Backend::textureResidentCount;
ContextMetricCount Backend::textureFramebufferCount;
ContextMetricCount Backend::textureResourceCount;
ContextMetricCount Backend::textureExternalCount;

ContextMetricSize  Backend::textureResidentGPUMemSize;
ContextMetricSize  Backend::textureFramebufferGPUMemSize;
ContextMetricSize  Backend::textureResourceGPUMemSize;
ContextMetricSize  Backend::textureExternalGPUMemSize;

ContextMetricCount Backend::texturePendingGPUTransferCount;
ContextMetricSize  Backend::texturePendingGPUTransferMemSize;

ContextMetricSize  Backend::textureResourcePopulatedGPUMemSize;
ContextMetricSize  Backend::textureResourceIdealGPUMemSize;

Size Context::getFreeGPUMemSize() {
    return Backend::freeGPUMemSize.getValue();
}

Size Context::getUsedGPUMemSize() {
    return getTextureGPUMemSize() + getBufferGPUMemSize();
};

uint32_t Context::getBufferGPUCount() {
    return Backend::bufferCount.getValue();
}

Context::Size Context::getBufferGPUMemSize() {
    return Backend::bufferGPUMemSize.getValue();
}

uint32_t Context::getTextureGPUCount() {
    return getTextureResidentGPUCount() + getTextureResourceGPUCount() + getTextureFramebufferGPUCount();
}
uint32_t Context::getTextureResidentGPUCount() {
    return Backend::textureResidentCount.getValue();
}
uint32_t Context::getTextureFramebufferGPUCount() {
    return Backend::textureFramebufferCount.getValue();
}
uint32_t Context::getTextureResourceGPUCount() {
    return Backend::textureResourceCount.getValue();
}
uint32_t Context::getTextureExternalGPUCount() {
    return Backend::textureExternalCount.getValue();
}

Size Context::getTextureGPUMemSize() {
    return getTextureResidentGPUMemSize() + getTextureResourceGPUMemSize() + getTextureFramebufferGPUMemSize();
}
Size Context::getTextureResidentGPUMemSize() {
    return Backend::textureResidentGPUMemSize.getValue();
}
Size Context::getTextureFramebufferGPUMemSize() {
    return Backend::textureFramebufferGPUMemSize.getValue();
}
Size Context::getTextureResourceGPUMemSize() {
    return Backend::textureResourceGPUMemSize.getValue();
}
Size Context::getTextureExternalGPUMemSize() {
    return Backend::textureExternalGPUMemSize.getValue();
}

uint32_t Context::getTexturePendingGPUTransferCount() {
    return Backend::texturePendingGPUTransferCount.getValue();
}

Size Context::getTexturePendingGPUTransferMemSize() {
    return Backend::texturePendingGPUTransferMemSize.getValue();
}

Size Context::getTextureResourcePopulatedGPUMemSize() {
    return Backend::textureResourcePopulatedGPUMemSize.getValue();
}

Size Context::getTextureResourceIdealGPUMemSize() {
    return Backend::textureResourceIdealGPUMemSize.getValue();
}


BatchPointer Context::acquireBatch(const char* name) {
    Batch* rawBatch = nullptr;
    {
        Lock lock(_batchPoolMutex);
        if (!_batchPool.empty()) {
            rawBatch = _batchPool.front();
            _batchPool.pop_front();
        }
    }
    if (!rawBatch) {
        rawBatch = new Batch();
    }
    rawBatch->setName(name);
    return BatchPointer(rawBatch, [this](Batch* batch) { releaseBatch(batch); });
}

void Context::releaseBatch(Batch* batch) {
    batch->clear();
    Lock lock(_batchPoolMutex);
    _batchPool.push_back(batch);
}

void gpu::doInBatch(const char* name,
                    const std::shared_ptr<gpu::Context>& context,
                    const std::function<void(Batch& batch)>& f) {
    auto batch = context->acquireBatch(name);
    f(*batch);
    context->appendFrameBatch(batch);
}

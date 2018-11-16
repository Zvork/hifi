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
#include <shaders/Shaders.h>

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

void Context::shutdown() {
    if (_backend) {
        _backend->shutdown();
        _backend.reset();
    }
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
    PROFILE_RANGE(render_gpu, __FUNCTION__);
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
    PROFILE_RANGE(render_gpu, __FUNCTION__);
    _backend->recycle();
}

void Context::consumeFrameUpdates(const FramePointer& frame) const {
    PROFILE_RANGE(render_gpu, __FUNCTION__);
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

PipelinePointer Context::createMipGenerationPipeline(const ShaderPointer& ps) {
    auto vs = gpu::Shader::createVertex(shader::gpu::vertex::DrawViewportQuadTransformTexcoord);
	static gpu::StatePointer state(new gpu::State());

	gpu::ShaderPointer program = gpu::Shader::createProgram(vs, ps);

	// Good to go add the brand new pipeline
	return gpu::Pipeline::create(program, state);
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

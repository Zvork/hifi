//
//  AntialiasingEffect.cpp
//  libraries/render-utils/src/
//
//  Created by Raffi Bedikian on 8/30/15
//  Copyright 2015 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#include "AntialiasingEffect.h"

#include <glm/gtc/random.hpp>

#include <SharedUtil.h>
#include <gpu/Context.h>
#include <shaders/Shaders.h>
#include <graphics/ShaderConstants.h>

#include "render-utils/ShaderConstants.h"
#include "StencilMaskPass.h"


namespace ru {
    using render_utils::slot::texture::Texture;
    using render_utils::slot::buffer::Buffer;
}

namespace gr {
    using graphics::slot::texture::Texture;
    using graphics::slot::buffer::Buffer;
}

gpu::PipelinePointer Antialiasing::_antialiasingPipeline;
gpu::PipelinePointer Antialiasing::_intensityPipeline;
gpu::PipelinePointer Antialiasing::_blendPipeline;
gpu::PipelinePointer Antialiasing::_debugBlendPipeline;

#if !ANTIALIASING_USE_TAA
#include "GeometryCache.h"
#include "ViewFrustum.h"
#include "DependencyManager.h"

#include "fxaa_vert.h"
#include "fxaa_frag.h"
#include "fxaa_blend_frag.h"


Antialiasing::Antialiasing() {
    _geometryId = DependencyManager::get<GeometryCache>()->allocateID();
}

Antialiasing::~Antialiasing() {
    auto geometryCache = DependencyManager::get<GeometryCache>();
    if (geometryCache) {
        geometryCache->releaseID(_geometryId);
    }
}

const gpu::PipelinePointer& Antialiasing::getAntialiasingPipeline() {
    if (!_antialiasingPipeline) {
        gpu::ShaderPointer program = gpu::Shader::createProgram(shader::render_utils::program::fxaa);
        gpu::StatePointer state = gpu::StatePointer(new gpu::State());

        state->setDepthTest(false, false, gpu::LESS_EQUAL);
        PrepareStencil::testNoAA(*state);

        // Good to go add the brand new pipeline
        _antialiasingPipeline = gpu::Pipeline::create(program, state);
    }

    return _antialiasingPipeline;
}

const gpu::PipelinePointer& Antialiasing::getBlendPipeline() {
    if (!_blendPipeline) {
        gpu::ShaderPointer program = gpu::Shader::createProgram(shader::render_utils::program::aa_blend);
        gpu::StatePointer state = gpu::StatePointer(new gpu::State());
        state->setDepthTest(false, false, gpu::LESS_EQUAL);
        PrepareStencil::testNoAA(*state);

        // Good to go add the brand new pipeline
        _blendPipeline = gpu::Pipeline::create(program, state);
    }
    return _blendPipeline;
}

void Antialiasing::run(const render::RenderContextPointer& renderContext, const gpu::FramebufferPointer& sourceBuffer) {
    assert(renderContext->args);
    assert(renderContext->args->hasViewFrustum());

    RenderArgs* args = renderContext->args;

    gpu::doInBatch("Antialiasing::run", args->_context, [&](gpu::Batch& batch) {
        batch.enableStereo(false);
        batch.setViewportTransform(args->_viewport);

        if (!_paramsBuffer) {
            _paramsBuffer = std::make_shared<gpu::Buffer>(sizeof(glm::vec4), nullptr);
        }

        {
            int width = args->_viewport.z; 
            int height = args->_viewport.w;
            if (_antialiasingBuffer && _antialiasingBuffer->getSize() != uvec2(width, height)) {
                _antialiasingBuffer.reset();
            }

            if (!_antialiasingBuffer) {
                // Link the antialiasing FBO to texture
                _antialiasingBuffer = gpu::FramebufferPointer(gpu::Framebuffer::create("antialiasing"));
                auto format = gpu::Element::COLOR_SRGBA_32;
                auto defaultSampler = gpu::Sampler(gpu::Sampler::FILTER_MIN_MAG_POINT);
                _antialiasingTexture = gpu::Texture::createRenderBuffer(format, width, height, gpu::Texture::SINGLE_MIP, defaultSampler);
                _antialiasingBuffer->setRenderBuffer(0, _antialiasingTexture);
                glm::vec2 fbExtent { args->_viewport.z, args->_viewport.w };
                glm::vec2 inverseFbExtent = 1.0f / fbExtent;
                _paramsBuffer->setSubData(0, glm::vec4(inverseFbExtent, 0.0, 0.0));
            }
        }


        batch.setSavedViewProjectionTransform(render::RenderEngine::TS_MAIN_VIEW);
        batch.setModelTransform(Transform());

        // FXAA step
        auto pipeline = getAntialiasingPipeline();
        batch.setResourceTexture(0, sourceBuffer->getRenderBuffer(0));
        batch.setFramebuffer(_antialiasingBuffer);
        batch.setPipeline(pipeline);
        batch.setUniformBuffer(0, _paramsBuffer);
        batch.draw(gpu::TRIANGLE_STRIP, 4);

        // Blend step
        batch.setResourceTexture(0, _antialiasingTexture);
        batch.setFramebuffer(sourceBuffer);
        batch.setPipeline(getBlendPipeline());
        batch.draw(gpu::TRIANGLE_STRIP, 4);
    });
}
#else
#include "RandomAndNoise.h"

#define TAA_JITTER_SEQUENCE_LENGTH 16

void AntialiasingSetupConfig::setIndex(int current) {
    _index = (current) % TAA_JITTER_SEQUENCE_LENGTH;
    emit dirty();
}

int AntialiasingSetupConfig::cycleStopPauseRun() {
    _state = (_state + 1) % 3;
    switch (_state) {
        case 0: {
            return none();
            break;
        }
        case 1: {
            return pause();
            break;
        }
        case 2:
        default: {
            return play();
            break;
        }
    }
    return _state;
}

int AntialiasingSetupConfig::prev() {
    setIndex(_index - 1);
    return _index;
}

int AntialiasingSetupConfig::next() {
    setIndex(_index + 1);
    return _index;
}

int AntialiasingSetupConfig::none() {
    _state = 0;
    stop = true;
    freeze = false;
    setIndex(-1);
    return _state;
}

int AntialiasingSetupConfig::pause() {
    _state = 1;
    stop = false;
    freeze = true;
    setIndex(0);
    return _state;
}

int AntialiasingSetupConfig::play() {
    _state = 2;
    stop = false;
    freeze = false;
    setIndex(0);
    return _state;
}

AntialiasingSetup::AntialiasingSetup() {
    _sampleSequence.reserve(TAA_JITTER_SEQUENCE_LENGTH + 1);
    // Fill in with jitter samples
    for (int i = 0; i < TAA_JITTER_SEQUENCE_LENGTH; i++) {
        _sampleSequence.emplace_back(glm::vec2(evaluateHalton<2>(i), evaluateHalton<3>(i)) - vec2(0.5f));
    }
}

void AntialiasingSetup::configure(const Config& config) {
    _isStopped = config.stop;
    _isFrozen = config.freeze;
    if (config.freeze) {
        _freezedSampleIndex = config.getIndex();
    }
    _scale = config.scale;
}

void AntialiasingSetup::run(const render::RenderContextPointer& renderContext) {
    assert(renderContext->args);
    if (!_isStopped) {
        RenderArgs* args = renderContext->args;

        gpu::doInBatch("AntialiasingSetup::run", args->_context, [&](gpu::Batch& batch) {
            auto offset = 0;
            auto count = _sampleSequence.size();
            if (_isFrozen) {
                count = 1;
                offset = _freezedSampleIndex;
            }
            batch.setProjectionJitterSequence(_sampleSequence.data() + offset, count);
            batch.setProjectionJitterScale(_scale);
        });
    }
}


Antialiasing::Antialiasing(bool isSharpenEnabled) : 
    _isSharpenEnabled{ isSharpenEnabled } {
}

Antialiasing::~Antialiasing() {
    _antialiasingBuffers.reset();
    _antialiasingTextures[0].reset();
    _antialiasingTextures[1].reset();
}

const gpu::PipelinePointer& Antialiasing::getAntialiasingPipeline() { 
    if (!_antialiasingPipeline) {
        gpu::ShaderPointer program = gpu::Shader::createProgram(shader::render_utils::program::taa);
        gpu::StatePointer state = gpu::StatePointer(new gpu::State());

        // Good to go add the brand new pipeline
        _antialiasingPipeline = gpu::Pipeline::create(program, state);
    }
    
    return _antialiasingPipeline;
}

const gpu::PipelinePointer& Antialiasing::getIntensityPipeline() {
    if (!_intensityPipeline) {
        gpu::ShaderPointer program = gpu::Shader::createProgram(shader::gpu::program::drawWhite);
        gpu::StatePointer state = gpu::StatePointer(new gpu::State());

        PrepareStencil::testNoAA(*state);

        // Good to go add the brand new pipeline
        _intensityPipeline = gpu::Pipeline::create(program, state);
    }

    return _intensityPipeline;
}

const gpu::PipelinePointer& Antialiasing::getBlendPipeline() {
    if (!_blendPipeline) {
        gpu::ShaderPointer program = gpu::Shader::createProgram(shader::render_utils::program::aa_blend);
        gpu::StatePointer state = gpu::StatePointer(new gpu::State());

        // Good to go add the brand new pipeline
        _blendPipeline = gpu::Pipeline::create(program, state);
    }
    return _blendPipeline;
}

const gpu::PipelinePointer& Antialiasing::getDebugBlendPipeline() {
    if (!_debugBlendPipeline) {
        gpu::ShaderPointer program = gpu::Shader::createProgram(shader::render_utils::program::taa_blend);
        gpu::StatePointer state = gpu::StatePointer(new gpu::State());
        PrepareStencil::testNoAA(*state);


        // Good to go add the brand new pipeline
        _debugBlendPipeline = gpu::Pipeline::create(program, state);
    }
    return _debugBlendPipeline;
}

void Antialiasing::configure(const Config& config) {
    _sharpen = config.sharpen * 0.25f;
    if (!_isSharpenEnabled) {
        _sharpen = 0.0f;
    }
    _params.edit().blend = config.blend * config.blend;
    _params.edit().covarianceGamma = config.covarianceGamma;

    _params.edit().setConstrainColor(config.constrainColor);
    _params.edit().setFeedbackColor(config.feedbackColor);

    _params.edit().debugShowVelocityThreshold = config.debugShowVelocityThreshold;

    _params.edit().regionInfo.x = config.debugX;
    _params.edit().regionInfo.z = config.debugFXAAX;

    _params.edit().setBicubicHistoryFetch(config.bicubicHistoryFetch);

    _params.edit().setDebug(config.debug);
    _params.edit().setShowDebugCursor(config.showCursorPixel);
    _params.edit().setDebugCursor(config.debugCursorTexcoord);
    _params.edit().setDebugOrbZoom(config.debugOrbZoom);

    _params.edit().setShowClosestFragment(config.showClosestFragment);
}


void Antialiasing::run(const render::RenderContextPointer& renderContext, const Inputs& inputs, Outputs& outputs) {
    assert(renderContext->args);
    
    RenderArgs* args = renderContext->args;

    auto& deferredFrameTransform = inputs.get0();
    const auto& deferredFrameBuffer = inputs.get1();
    const auto& sourceBuffer = deferredFrameBuffer->getLightingFramebuffer();
    const auto& linearDepthBuffer = inputs.get2();
    const auto& velocityTexture = deferredFrameBuffer->getDeferredVelocityTexture();
    
    int width = sourceBuffer->getWidth();
    int height = sourceBuffer->getHeight();

    if (_antialiasingBuffers && _antialiasingBuffers->get(0) && _antialiasingBuffers->get(0)->getSize() != uvec2(width, height)) {
        _antialiasingBuffers.reset();
        _antialiasingTextures[0].reset();
        _antialiasingTextures[1].reset();
    }

    if (!_antialiasingBuffers || !_intensityFramebuffer) {
        std::vector<gpu::FramebufferPointer> antiAliasingBuffers;
        // Link the antialiasing FBO to texture
        auto format = gpu::Element(gpu::VEC4, gpu::HALF, gpu::RGBA);
        auto defaultSampler = gpu::Sampler(gpu::Sampler::FILTER_MIN_MAG_LINEAR, gpu::Sampler::WRAP_CLAMP);
        for (int i = 0; i < 2; i++) {
            antiAliasingBuffers.emplace_back(gpu::Framebuffer::create("antialiasing"));
            const auto& antiAliasingBuffer = antiAliasingBuffers.back();
            _antialiasingTextures[i] = gpu::Texture::createRenderBuffer(format, width, height, gpu::Texture::SINGLE_MIP, defaultSampler);
            antiAliasingBuffer->setRenderBuffer(0, _antialiasingTextures[i]);
        }
        _antialiasingBuffers = std::make_shared<gpu::FramebufferSwapChain>(antiAliasingBuffers);

        _intensityTexture = gpu::Texture::createRenderBuffer(gpu::Element::COLOR_R_8, width, height, gpu::Texture::SINGLE_MIP, defaultSampler);
        _intensityFramebuffer = gpu::FramebufferPointer(gpu::Framebuffer::create("taaIntensity"));
        _intensityFramebuffer->setRenderBuffer(0, _intensityTexture);
        _intensityFramebuffer->setStencilBuffer(deferredFrameBuffer->getDeferredFramebuffer()->getDepthStencilBuffer(), deferredFrameBuffer->getDeferredFramebuffer()->getDepthStencilBufferFormat());
    }

    outputs = _intensityTexture;

    gpu::doInBatch("Antialiasing::run", args->_context, [&](gpu::Batch& batch) {
        PROFILE_RANGE_BATCH(batch, "TAA");

        batch.enableStereo(false);
        batch.setViewportTransform(args->_viewport);

        // Set the intensity buffer to 1 except when the stencil is masked as NoAA, where it should be 0
        // This is a bit of a hack as it is not possible and not portable to use the stencil value directly
        // as a texture
        batch.setFramebuffer(_intensityFramebuffer);
        batch.clearColorFramebuffer(gpu::Framebuffer::BUFFER_COLOR0, gpu::Vec4(0.0f));
        batch.setResourceTexture(0, nullptr);
        batch.setPipeline(getIntensityPipeline());
        batch.draw(gpu::TRIANGLE_STRIP, 4);

        // TAA step
        batch.setResourceFramebufferSwapChainTexture(ru::Texture::TaaHistory, _antialiasingBuffers, 0);
        batch.setResourceTexture(ru::Texture::TaaSource, sourceBuffer->getRenderBuffer(0));
        batch.setResourceTexture(ru::Texture::TaaVelocity, velocityTexture);
        batch.setResourceTexture(ru::Texture::TaaIntensity, _intensityTexture);
        // This is only used during debug
        batch.setResourceTexture(ru::Texture::TaaDepth, linearDepthBuffer->getLinearDepthTexture());

        batch.setUniformBuffer(ru::Buffer::TaaParams, _params);
        batch.setUniformBuffer(ru::Buffer::DeferredFrameTransform, deferredFrameTransform->getFrameTransformBuffer());
        
        batch.setFramebufferSwapChain(_antialiasingBuffers, 1);
        batch.setPipeline(getAntialiasingPipeline());
        batch.draw(gpu::TRIANGLE_STRIP, 4);

        // Blend step
        batch.setResourceTexture(ru::Texture::TaaSource, nullptr);

        batch.setFramebuffer(sourceBuffer);
        if (_params->isDebug()) {
            batch.setPipeline(getDebugBlendPipeline());
            batch.setResourceFramebufferSwapChainTexture(ru::Texture::TaaNext, _antialiasingBuffers, 1);
        }  else {
            batch.setPipeline(getBlendPipeline());
            // Must match the binding point in the aa_blend.slf shader
            batch.setResourceFramebufferSwapChainTexture(0, _antialiasingBuffers, 1);
            // Disable sharpen if FXAA
            if (!_blendParamsBuffer) {
                _blendParamsBuffer = std::make_shared<gpu::Buffer>(sizeof(glm::vec4), nullptr);
            }
            _blendParamsBuffer->setSubData(0, _sharpen * _params.get().regionInfo.z);
            batch.setUniformBuffer(0, _blendParamsBuffer);
        }
        batch.draw(gpu::TRIANGLE_STRIP, 4);
        batch.advance(_antialiasingBuffers);
        
        batch.setUniformBuffer(ru::Buffer::TaaParams, nullptr);
        batch.setUniformBuffer(ru::Buffer::DeferredFrameTransform, nullptr);

        batch.setResourceTexture(ru::Texture::TaaDepth, nullptr);
        batch.setResourceTexture(ru::Texture::TaaHistory, nullptr);
        batch.setResourceTexture(ru::Texture::TaaVelocity, nullptr);
        batch.setResourceTexture(ru::Texture::TaaNext, nullptr);

        // Reset jitter sequence
        batch.setProjectionJitterSequence(nullptr, 0);
    });
}

#endif

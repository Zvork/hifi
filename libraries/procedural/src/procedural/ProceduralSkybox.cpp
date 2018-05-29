//
//  ProceduralSkybox.cpp
//  libraries/procedural/src/procedural
//
//  Created by Sam Gateau on 9/21/2015.
//  Copyright 2015 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#include "ProceduralSkybox.h"


#include <gpu/Batch.h>
#include <gpu/Context.h>
#include <ViewFrustum.h>

#include <graphics/skybox_vert.h>
#include <graphics/skybox_frag.h>
#include <graphics/skybox_fwd_vert.h>
#include <graphics/skybox_fwd_frag.h>

ProceduralSkybox::ProceduralSkybox() : graphics::Skybox() {
    _procedural.setVertexSource( skybox_vert::getSource() );
    _procedural.setOpaqueFragmentSource( skybox_frag::getSource() );
    // Adjust the pipeline state for background using the stencil test
    _procedural.setDoesFade(false);
    // Must match PrepareStencil::STENCIL_BACKGROUND
    const int8_t STENCIL_BACKGROUND = 0;
    _procedural._opaqueState->setStencilTest(true, 0xFF, gpu::State::StencilTest(STENCIL_BACKGROUND, 0xFF, gpu::EQUAL,
        gpu::State::STENCIL_OP_KEEP, gpu::State::STENCIL_OP_KEEP, gpu::State::STENCIL_OP_KEEP));
}

bool ProceduralSkybox::empty() {
    return !_procedural.isEnabled() && Skybox::empty();
}

void ProceduralSkybox::clear() {
    // Parse and prepare a procedural with no shaders to release textures
    parse(QString());
    _procedural.isReady();

    Skybox::clear();
}

void ProceduralSkybox::render(gpu::Batch& batch, bool isDeferred, const ViewFrustum& frustum, uint xformSlot) const {
    if (_procedural.isReady()) {
        if (_isDeferred != isDeferred) {
            // Choose correct shader source. This is propably not the optimal way, especially if the procedural
            // skybox is drawn in the same frame in both deferred AND forward.
            if (isDeferred) {
                _procedural.setVertexSource( skybox_vert::getSource() );
                _procedural.setOpaqueFragmentSource( skybox_frag::getSource() );
            } else {
                _procedural.setVertexSource( skybox_fwd_vert::getSource() );
                _procedural.setOpaqueFragmentSource( skybox_fwd_frag::getSource() );
            }
            _isDeferred = isDeferred;
        }
        ProceduralSkybox::render(batch, isDeferred, frustum, (*this), xformSlot);
    } else {
        Skybox::render(batch, isDeferred, frustum, xformSlot);
    }
}

void ProceduralSkybox::render(gpu::Batch& batch, bool isDeferred, const ViewFrustum& viewFrustum, const ProceduralSkybox& skybox, uint xformSlot) {
    glm::mat4 projMat;
    viewFrustum.evalProjectionMatrix(projMat);

    Transform viewTransform;
    viewFrustum.evalViewTransform(viewTransform);
    batch.setProjectionTransform(projMat);
    batch.setViewTransform(viewTransform);
    // This is needed if we want to have motion vectors on the sky
    batch.saveViewProjectionTransform(xformSlot);
    batch.setModelTransform(Transform());  // only for Mac

    auto& procedural = skybox._procedural;
    procedural.prepare(batch, glm::vec3(0), glm::vec3(1), glm::quat());
    auto textureSlot = procedural.getOpaqueShader()->getTextures().findLocation("cubeMap");
    auto bufferSlot = procedural.getOpaqueShader()->getUniformBuffers().findLocation("skyboxBuffer");
    skybox.prepare(batch, textureSlot, bufferSlot);
    batch.draw(gpu::TRIANGLE_STRIP, 4);
}


//
//  Skybox.cpp
//  libraries/graphics/src/graphics
//
//  Created by Sam Gateau on 5/4/2015.
//  Copyright 2015 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#include "Skybox.h"


#include <gpu/Batch.h>
#include <gpu/Context.h>
#include <ViewFrustum.h>

#include "skybox_vert.h"
#include "skybox_frag.h"
#include "skybox_fwd_vert.h"
#include "skybox_fwd_frag.h"

using namespace graphics;

Skybox::Skybox() {
    Schema schema;
    _schemaBuffer = gpu::BufferView(std::make_shared<gpu::Buffer>(sizeof(Schema), (const gpu::Byte*) &schema));
}

void Skybox::setColor(const Color& color) {
    _schemaBuffer.edit<Schema>().color = color;
    _empty = false;
}

void Skybox::setCubemap(const gpu::TexturePointer& cubemap) {
    _cubemap = cubemap;
    if (cubemap) {
        _empty = false;
    }
}

void Skybox::setOrientation(const glm::quat& orientation) {
    // The zone rotations need to be negated
    _orientation = orientation;
    _orientation.w = -_orientation.w;
}

void Skybox::updateSchemaBuffer() const {
    auto blend = 0.0f;
    if (getCubemap() && getCubemap()->isDefined()) {
        blend = 0.5f;

        // If pitch black neutralize the color
        if (glm::all(glm::equal(getColor(), glm::vec3(0.0f)))) {
            blend = 1.0f;
        }
    }

    if (blend != _schemaBuffer.get<Schema>().blend) {
        _schemaBuffer.edit<Schema>().blend = blend;
    }
}

void Skybox::clear() {
    _schemaBuffer.edit<Schema>().color = vec3(0);
    _cubemap = nullptr;
    _empty = true;
}

void Skybox::prepare(gpu::Batch& batch, int textureSlot, int bufferSlot) const {
    if (bufferSlot > -1) {
        batch.setUniformBuffer(bufferSlot, _schemaBuffer);
    }

    if (textureSlot > -1) {
        gpu::TexturePointer skymap = getCubemap();
        // FIXME: skymap->isDefined may not be threadsafe
        if (skymap && skymap->isDefined()) {
            batch.setResourceTexture(textureSlot, skymap);
        }
    }
}

void Skybox::render(gpu::Batch& batch, bool isDeferred, const ViewFrustum& frustum, uint xformSlot) const {
    updateSchemaBuffer();
    Skybox::render(batch, isDeferred, frustum, (*this), xformSlot);
}

void Skybox::render(gpu::Batch& batch, bool isDeferred, const ViewFrustum& viewFrustum, const Skybox& skybox, uint xformSlot) {
    // Create the static shared elements used to render the skybox
    static gpu::BufferPointer theConstants;
    static gpu::StatePointer theState;
    static gpu::PipelinePointer theForwardPipeline;
    static gpu::PipelinePointer theDeferredPipeline;

    if (theState == nullptr) {
        theState = std::make_shared<gpu::State>();
        // Must match PrepareStencil::STENCIL_BACKGROUND
        const int8_t STENCIL_BACKGROUND = 0;
        theState->setStencilTest(true, 0xFF,
                                 gpu::State::StencilTest(STENCIL_BACKGROUND, 0xFF, gpu::EQUAL, gpu::State::STENCIL_OP_KEEP,
                                                         gpu::State::STENCIL_OP_KEEP, gpu::State::STENCIL_OP_KEEP));
    }

    gpu::PipelinePointer pipeline;

    if (isDeferred) {
        if (theDeferredPipeline == nullptr) {
            auto skyVS = skybox_vert::getShader();
            auto skyFS = skybox_frag::getShader();
            auto skyShader = gpu::Shader::createProgram(skyVS, skyFS);

            batch.runLambda([skyShader] {
                gpu::Shader::BindingSet bindings;
                bindings.insert(gpu::Shader::Binding(std::string("cubeMap"), SKYBOX_SKYMAP_SLOT));
                bindings.insert(gpu::Shader::Binding(std::string("skyboxBuffer"), SKYBOX_CONSTANTS_SLOT));
                if (!gpu::Shader::makeProgram(*skyShader, bindings)) {
                }
            });

            theDeferredPipeline = gpu::Pipeline::create(skyShader, theState);
        }
        pipeline = theDeferredPipeline;
    } else {
        if (theForwardPipeline == nullptr) {
            auto skyVS = skybox_fwd_vert::getShader();
            auto skyFS = skybox_fwd_frag::getShader();
            auto skyShader = gpu::Shader::createProgram(skyVS, skyFS);

            batch.runLambda([skyShader] {
                gpu::Shader::BindingSet bindings;
                bindings.insert(gpu::Shader::Binding(std::string("cubeMap"), SKYBOX_SKYMAP_SLOT));
                bindings.insert(gpu::Shader::Binding(std::string("skyboxBuffer"), SKYBOX_CONSTANTS_SLOT));
                if (!gpu::Shader::makeProgram(*skyShader, bindings)) {
                }
            });

            theForwardPipeline = gpu::Pipeline::create(skyShader, theState);
        }
        pipeline = theForwardPipeline;
    }

    // Render
    glm::mat4 projMat;
    viewFrustum.evalProjectionMatrix(projMat);

    Transform viewTransform;
    viewFrustum.evalViewTransform(viewTransform);

    // Orientate view transform to be relative to zone
    viewTransform.setRotation(skybox.getOrientation() * viewTransform.getRotation());

    batch.setProjectionTransform(projMat);
    batch.setViewTransform(viewTransform);
    // This is needed if we want to have motion vectors on the sky
    batch.saveViewProjectionTransform(xformSlot);
    batch.setModelTransform(Transform()); // only for Mac

    batch.setPipeline(pipeline);
    skybox.prepare(batch);
    batch.draw(gpu::TRIANGLE_STRIP, 4);

    batch.setResourceTexture(SKYBOX_SKYMAP_SLOT, nullptr);
}

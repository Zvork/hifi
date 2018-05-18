//
//  DeferredFrameTransform.cpp
//  libraries/render-utils/src/
//
//  Created by Sam Gateau 6/3/2016.
//  Copyright 2016 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#include "DeferredFrameTransform.h"

#include "gpu/Context.h"
#include "render/Engine.h"

DeferredFrameTransform::DeferredFrameTransform() {
    FrameTransform frameTransform;
    _frameTransformBuffer = gpu::BufferView(std::make_shared<gpu::Buffer>(sizeof(FrameTransform), (const gpu::Byte*) &frameTransform));
}

void DeferredFrameTransform::update(RenderArgs* args) {

    // Update the depth info with near and far (same for stereo)
    auto nearZ = args->getViewFrustum().getNearClip();
    auto farZ = args->getViewFrustum().getFarClip();

    auto& frameTransformBuffer = _frameTransformBuffer.edit<FrameTransform>();
    frameTransformBuffer.depthInfo = glm::vec4(nearZ*farZ, farZ - nearZ, -farZ, 0.0f);

    frameTransformBuffer.pixelInfo = args->_viewport;

    //_parametersBuffer.edit<Parameters>()._ditheringInfo.y += 0.25f;

    Transform cameraTransform;
    args->getViewFrustum().evalViewTransform(cameraTransform);
    cameraTransform.getMatrix(frameTransformBuffer.invView);
    cameraTransform.getInverseMatrix(frameTransformBuffer.view);

    args->getViewFrustum().evalProjectionMatrix(frameTransformBuffer.projectionMono);

    // Running in stereo ?
    bool isStereo = args->isStereo();
    if (!isStereo) {
        frameTransformBuffer.projectionUnjittered[0] = frameTransformBuffer.projectionMono;
        frameTransformBuffer.invProjectionUnjittered[0] = glm::inverse(frameTransformBuffer.projectionUnjittered[0]);

        frameTransformBuffer.stereoInfo = glm::vec4(0.0f, (float)args->_viewport.z, 0.0f, 0.0f);
        frameTransformBuffer.invPixelInfo = glm::vec4(1.0f / args->_viewport.z, 1.0f / args->_viewport.w, 0.0f, 0.0f);

		frameTransformBuffer.projection[0] = frameTransformBuffer.projectionUnjittered[0];
//		frameTransformBuffer.projection[0][2][0] += jitter.x;
//		frameTransformBuffer.projection[0][2][1] += jitter.y;
        frameTransformBuffer.invProjection[0] = glm::inverse(frameTransformBuffer.projection[0]);
    } else {

        mat4 projMats[2];
        mat4 eyeViews[2];
        args->_context->getStereoProjections(projMats);
        args->_context->getStereoViews(eyeViews);

        for (int i = 0; i < 2; i++) {
            // Compose the mono Eye space to Stereo clip space Projection Matrix
            auto sideViewMat = projMats[i] * eyeViews[i];
            frameTransformBuffer.projectionUnjittered[i] = sideViewMat;
            frameTransformBuffer.invProjectionUnjittered[i] = glm::inverse(sideViewMat);

			frameTransformBuffer.projection[i] = frameTransformBuffer.projectionUnjittered[i];
//			frameTransformBuffer.projection[i][2][0] += jitter.x;
//			frameTransformBuffer.projection[i][2][1] += jitter.y;
			frameTransformBuffer.invProjection[i] = glm::inverse(frameTransformBuffer.projection[i]);
		}

        frameTransformBuffer.stereoInfo = glm::vec4(1.0f, (float)(args->_viewport.z >> 1), 0.0f, 1.0f);
        frameTransformBuffer.invPixelInfo = glm::vec4(1.0f / (float)(args->_viewport.z >> 1), 1.0f / args->_viewport.w, 0.0f, 0.0f);
    }
}

void GenerateDeferredFrameTransform::run(const render::RenderContextPointer& renderContext, Output& frameTransform) {
    if (!frameTransform) {
        frameTransform = std::make_shared<DeferredFrameTransform>();
    }
    frameTransform->update(renderContext->args);
    RenderArgs* args = renderContext->args;

    gpu::doInBatch("GenerateDeferredFrameTransform::run", args->_context, [&](gpu::Batch& batch) {
        args->_batch = &batch;

        // Setup camera, projection and viewport for all items
        batch.setViewportTransform(args->_viewport);
        batch.setStateScissorRect(args->_viewport);

        glm::mat4 projMat;
        Transform viewMat;
        args->getViewFrustum().evalProjectionMatrix(projMat);
        args->getViewFrustum().evalViewTransform(viewMat);
        batch.setProjectionTransform(projMat);
        batch.setViewTransform(viewMat);
        // This is the main view / projection transform that will be reused later on
        batch.saveViewProjectionTransform(0);
    });
}

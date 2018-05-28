//
//  BackgroundStage.cpp
//
//  Created by Sam Gateau on 5/9/2017.
//  Copyright 2015 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#include "BackgroundStage.h"

#include "DeferredLightingEffect.h"

#include <gpu/Context.h>

std::string BackgroundStage::_stageName{ "BACKGROUND_STAGE" };
const BackgroundStage::Index BackgroundStage::INVALID_INDEX{ render::indexed_container::INVALID_INDEX };

BackgroundStage::Index BackgroundStage::findBackground(const BackgroundPointer& background) const {
    auto found = _backgroundMap.find(background);
    if (found != _backgroundMap.end()) {
        return INVALID_INDEX;
    } else {
        return (*found).second;
    }
}

BackgroundStage::Index BackgroundStage::addBackground(const BackgroundPointer& background) {
    auto found = _backgroundMap.find(background);
    if (found == _backgroundMap.end()) {
        auto backgroundId = _backgrounds.newElement(background);
        // Avoid failing to allocate a background, just pass
        if (backgroundId != INVALID_INDEX) {
            // Insert the background and its index in the reverse map
            _backgroundMap.insert(BackgroundMap::value_type(background, backgroundId));
        }
        return backgroundId;
    } else {
        return (*found).second;
    }
}

BackgroundStage::BackgroundPointer BackgroundStage::removeBackground(Index index) {
    BackgroundPointer removed = _backgrounds.freeElement(index);

    if (removed) {
        _backgroundMap.erase(removed);
    }
    return removed;
}

void DrawBackgroundStage::run(const render::RenderContextPointer& renderContext, const Inputs& inputs) {
    const auto& lightingModel = inputs.get0();
    if (!lightingModel->isBackgroundEnabled()) {
        return;
    }

    // Background rendering decision
    auto backgroundStage = renderContext->_scene->getStage<BackgroundStage>();
    assert(backgroundStage);

    graphics::SunSkyStagePointer background;
    graphics::SkyboxPointer skybox;
    if (backgroundStage->_currentFrame._backgrounds.size()) {
        auto backgroundId = backgroundStage->_currentFrame._backgrounds.front();
        auto background = backgroundStage->getBackground(backgroundId);
        if (background) {
            skybox = background->getSkybox();
        }
    }
    /*  auto backgroundMode = skyStage->getBackgroundMode();

    switch (backgroundMode) {
    case graphics::SunSkyStage::SKY_DEFAULT: {
        auto scene = DependencyManager::get<SceneScriptingInterface>()->getStage();
        auto sceneKeyLight = scene->getKeyLight();

        scene->setSunModelEnable(false);
        sceneKeyLight->setColor(ColorUtils::toVec3(KeyLightPropertyGroup::DEFAULT_KEYLIGHT_COLOR));
        sceneKeyLight->setIntensity(KeyLightPropertyGroup::DEFAULT_KEYLIGHT_INTENSITY);
        sceneKeyLight->setAmbientIntensity(KeyLightPropertyGroup::DEFAULT_KEYLIGHT_AMBIENT_INTENSITY);
        sceneKeyLight->setDirection(KeyLightPropertyGroup::DEFAULT_KEYLIGHT_DIRECTION);
        // fall through: render a skybox (if available), or the defaults (if requested)
    }

    case graphics::SunSkyStage::SKY_BOX: {*/
    if (skybox && !skybox->empty()) {

        PerformanceTimer perfTimer("skybox");
        auto args = renderContext->args;
        const auto& deferredFrameBuffer = inputs.get1();
        const auto& finalFrameBuffer = inputs.get2();


        gpu::doInBatch("DrawBackgroundStage::run", args->_context, [&](gpu::Batch& batch) {
            args->_batch = &batch;

            batch.setFramebuffer(deferredFrameBuffer->getDeferredFramebuffer());
            batch.enableSkybox(true);

            batch.setViewportTransform(args->_viewport);
            batch.setStateScissorRect(args->_viewport);
            batch.setProjectionJitterEnabled(true);

            skybox->render(batch, args->getViewFrustum(), render::RenderEngine::TS_BACKGROUND_VIEW);

            // Restore final framebuffer
            batch.setFramebuffer(finalFrameBuffer);
        });
        args->_batch = nullptr;

        // break;
    }
    // fall through: render defaults (if requested)
    //    }
    /*
    case graphics::SunSkyStage::SKY_DEFAULT_AMBIENT_TEXTURE: {
        if (Menu::getInstance()->isOptionChecked(MenuOption::DefaultSkybox)) {
            auto scene = DependencyManager::get<SceneScriptingInterface>()->getStage();
            auto sceneKeyLight = scene->getKeyLight();
            auto defaultSkyboxAmbientTexture = qApp->getDefaultSkyboxAmbientTexture();
            if (defaultSkyboxAmbientTexture) {
                sceneKeyLight->setAmbientSphere(defaultSkyboxAmbientTexture->getIrradiance());
                sceneKeyLight->setAmbientMap(defaultSkyboxAmbientTexture);
            }
            // fall through: render defaults skybox
        } else {
            break;
        }
    }
    */
}

BackgroundStageSetup::BackgroundStageSetup() {
}

void BackgroundStageSetup::run(const render::RenderContextPointer& renderContext) {
    auto stage = renderContext->_scene->getStage(BackgroundStage::getName());
    if (!stage) {
        renderContext->_scene->resetStage(BackgroundStage::getName(), std::make_shared<BackgroundStage>());
    }
}

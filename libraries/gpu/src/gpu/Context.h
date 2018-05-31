//
//  Context.h
//  interface/src/gpu
//
//  Created by Sam Gateau on 10/27/2014.
//  Copyright 2014 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#ifndef hifi_gpu_Context_h
#define hifi_gpu_Context_h

#include <assert.h>
#include <mutex>

#include "Texture.h"
#include "Pipeline.h"
#include "Frame.h"
#include "Backend.h"

namespace gpu {

class Context {
public:
    using Size = Resource::Size;
    typedef BackendPointer (*CreateBackend)();
    typedef bool (*MakeProgram)(Shader& shader, const Shader::BindingSet& bindings, const Shader::CompilationHandler& handler);

    // This one call must happen before any context is created or used (Shader::MakeProgram) in order to setup the Backend and any singleton data needed
    template <class T>
    static void init() {
        std::call_once(_initialized, [] {
            _createBackendCallback = T::createBackend;
            _makeProgramCallback = T::makeProgram;
            T::init();
        });
    }

    Context();
    ~Context();

    void shutdown();
    const std::string& getBackendVersion() const;

    void beginFrame(const glm::mat4& renderView = glm::mat4(), const glm::mat4& renderPose = glm::mat4());
    void appendFrameBatch(const BatchPointer& batch);
    FramePointer endFrame();

    BatchPointer acquireBatch(const char* name = nullptr);
    void releaseBatch(Batch* batch);

    // MUST only be called on the rendering thread
    //
    // Handle any pending operations to clean up (recycle / deallocate) resources no longer in use
    void recycle() const;

    // MUST only be called on the rendering thread
    //
    // Execute a batch immediately, rather than as part of a frame
    void executeBatch(Batch& batch) const;

    // MUST only be called on the rendering thread
    //
    // Executes a frame, applying any updates contained in the frame batches to the rendering
    // thread shadow copies.  Either executeFrame or consumeFrameUpdates MUST be called on every frame
    // generated, IN THE ORDER they were generated.
    void executeFrame(const FramePointer& frame) const;

    // MUST only be called on the rendering thread.
    //
    // Consuming a frame applies any updates queued from the recording thread and applies them to the
    // shadow copy used by the rendering thread.
    //
    // EVERY frame generated MUST be consumed, regardless of whether the frame is actually executed,
    // or the buffer shadow copies can become unsynced from the recording thread copies.
    //
    // Consuming a frame is idempotent, as the frame encapsulates the updates and clears them out as
    // it applies them, so calling it more than once on a given frame will have no effect after the
    // first time
    //
    //
    // This is automatically called by executeFrame, so you only need to call it if you
    // have frames you aren't going to otherwise execute, for instance when a display plugin is
    // being disabled, or in the null display plugin where no rendering actually occurs
    void consumeFrameUpdates(const FramePointer& frame) const;

    const BackendPointer& getBackend() const { return _backend; }

    void enableStereo(bool enable = true);
    bool isStereo();
    void setStereoProjections(const mat4 eyeProjections[2]);
    void setStereoViews(const mat4 eyeViews[2]);
    void getStereoProjections(mat4* eyeProjections) const;
    void getStereoViews(mat4* eyeViews) const;

    // Downloading the Framebuffer is a synchronous action that is not efficient.
    // It s here for convenience to easily capture a snapshot
    void downloadFramebuffer(const FramebufferPointer& srcFramebuffer, const Vec4i& region, QImage& destImage);

    // Repporting stats of the context
    void resetStats() const;
    void getStats(ContextStats& stats) const;

    // Same as above but grabbed at every end of a frame
    void getFrameStats(ContextStats& stats) const;

    double getFrameTimerGPUAverage() const;
    double getFrameTimerBatchAverage() const;

    static Size getFreeGPUMemSize();
    static Size getUsedGPUMemSize();

    static uint32_t getBufferGPUCount();
    static Size getBufferGPUMemSize();

    static uint32_t getTextureGPUCount();
    static uint32_t getTextureResidentGPUCount();
    static uint32_t getTextureFramebufferGPUCount();
    static uint32_t getTextureResourceGPUCount();
    static uint32_t getTextureExternalGPUCount();

    static Size getTextureGPUMemSize();
    static Size getTextureResidentGPUMemSize();
    static Size getTextureFramebufferGPUMemSize();
    static Size getTextureResourceGPUMemSize();
    static Size getTextureExternalGPUMemSize();

    static uint32_t getTexturePendingGPUTransferCount();
    static Size getTexturePendingGPUTransferMemSize();

    static Size getTextureResourcePopulatedGPUMemSize();
    static Size getTextureResourceIdealGPUMemSize();

protected:
    Context(const Context& context);

    std::shared_ptr<Backend> _backend;
    std::mutex _batchPoolMutex;
    std::list<Batch*> _batchPool;
    bool _frameActive{ false };
    FramePointer _currentFrame;
    RangeTimerPointer _frameRangeTimer;
    StereoState _stereo;

    // Sampled at the end of every frame, the stats of all the counters
    mutable ContextStats _frameStats;

    // This function can only be called by "static Shader::makeProgram()"
    // makeProgramShader(...) make a program shader ready to be used in a Batch.
    // It compiles the sub shaders, link them and defines the Slots and their bindings.
    // If the shader passed is not a program, nothing happens.
    static bool makeProgram(Shader& shader, const Shader::BindingSet& bindings, const Shader::CompilationHandler& handler);

    static CreateBackend _createBackendCallback;
    static MakeProgram _makeProgramCallback;
    static std::once_flag _initialized;

    friend class Shader;
    friend class Backend;
};
typedef std::shared_ptr<Context> ContextPointer;

void doInBatch(const char* name, const std::shared_ptr<gpu::Context>& context, const std::function<void(Batch& batch)>& f);

};  // namespace gpu

#endif

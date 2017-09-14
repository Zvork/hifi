//
//  CubeMap.cpp
//  image/src/image
//
//  Created by Olivier Prat on 09/14/2017.
//  Copyright 2017 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#include "CubeMap.h"

#include <nvtt/nvtt.h>

// Necessary for M_PI definition
#include <qmath.h>

extern void compressHDRMip(gpu::Texture* texture, const nvtt::Surface& surface, int face, int mipLevel);
extern void convertQImageToVec4s(const QImage& image, const gpu::Element format, std::vector<glm::vec4>& vec4s);

static const glm::vec3 faceNormals[6] = {
    glm::vec3(1, 0, 0),
    glm::vec3(-1, 0, 0),
    glm::vec3(0, 1, 0),
    glm::vec3(0, -1, 0),
    glm::vec3(0, 0, 1),
    glm::vec3(0, 0, -1),
};

static glm::vec3 texelDirection(uint face, uint x, uint y, int edgeLength) {
    float u, v;

    // Transform x,y to [-1, 1] range, offset by 0.5 to point to texel center.
    u = (float(x) + 0.5f) * (2.0f / edgeLength) - 1.0f;
    v = (float(y) + 0.5f) * (2.0f / edgeLength) - 1.0f;

    assert(u >= -1.0f && u <= 1.0f);
    assert(v >= -1.0f && v <= 1.0f);

    glm::vec3 n;

    switch (face) {
    case gpu::Texture::CUBE_FACE_RIGHT_POS_X:
        n.x = 1;
        n.y = -v;
        n.z = -u;
        break;
    case gpu::Texture::CUBE_FACE_LEFT_NEG_X:
        n.x = -1;
        n.y = -v;
        n.z = u;
        break;
    case gpu::Texture::CUBE_FACE_TOP_POS_Y:
        n.x = u;
        n.y = 1;
        n.z = v;
        break;
    case gpu::Texture::CUBE_FACE_BOTTOM_NEG_Y:
        n.x = u;
        n.y = -1;
        n.z = -v;
        break;
    case gpu::Texture::CUBE_FACE_BACK_POS_Z:
        n.x = u;
        n.y = -v;
        n.z = 1;
        break;
    case gpu::Texture::CUBE_FACE_FRONT_NEG_Z:
        n.x = -u;
        n.y = -v;
        n.z = -1;
        break;
    }

    return glm::normalize(n);
}

// Solid angle of an axis aligned quad from (0,0,1) to (x,y,1)
// See: http://www.fizzmoll11.com/thesis/ for a derivation of this formula.
static float areaElement(float x, float y) {
    return atan2(x*y, sqrtf(x*x + y*y + 1));
}

// Solid angle of a hemicube texel.
static float solidAngleTerm(uint x, uint y, float inverseEdgeLength) {
    // Transform x,y to [-1, 1] range, offset by 0.5 to point to texel center.
    float u = (float(x) + 0.5f) * (2 * inverseEdgeLength) - 1.0f;
    float v = (float(y) + 0.5f) * (2 * inverseEdgeLength) - 1.0f;
    assert(u >= -1.0f && u <= 1.0f);
    assert(v >= -1.0f && v <= 1.0f);

    // Exact solid angle:
    float x0 = u - inverseEdgeLength;
    float y0 = v - inverseEdgeLength;
    float x1 = u + inverseEdgeLength;
    float y1 = v + inverseEdgeLength;
    float solidAngle = areaElement(x0, y0) - areaElement(x0, y1) - areaElement(x1, y0) + areaElement(x1, y1);
    assert(solidAngle > 0.0f);

    return solidAngle;
}

namespace image {

    struct TexelTable {

        TexelTable(uint edgeLength) : size(edgeLength) {

            uint hsize = size / 2;

            // Allocate a small solid angle table that takes into account cube map symmetry.
            solidAngleArray.resize(hsize * hsize);

            for (uint y = 0; y < hsize; y++) {
                for (uint x = 0; x < hsize; x++) {
                    solidAngleArray[y * hsize + x] = solidAngleTerm(hsize + x, hsize + y, 1.0f / edgeLength);
                }
            }

            directionArray.resize(size*size * 6);

            for (uint f = 0; f < 6; f++) {
                for (uint y = 0; y < size; y++) {
                    for (uint x = 0; x < size; x++) {
                        directionArray[(f * size + y) * size + x] = texelDirection(f, x, y, edgeLength);
                    }
                }
            }
        }

        float solidAngle(uint f, uint x, uint y) const {
            uint hsize = size / 2;
            if (x >= hsize) x -= hsize;
            else if (x < hsize) x = hsize - x - 1;
            if (y >= hsize) y -= hsize;
            else if (y < hsize) y = hsize - y - 1;

            return solidAngleArray[y * hsize + x];
        }

        const glm::vec3 & direction(uint f, uint x, uint y) const {
            assert(f < 6 && x < size && y < size);
            return directionArray[(f * size + y) * size + x];
        }

        uint size;
        std::vector<float> solidAngleArray;
        std::vector<glm::vec3> directionArray;
    };
}

void compressHDRCubeMap(gpu::Texture* texture, const nvtt::CubeSurface& cubeMap, int mipLevel) {
    for (auto i = 0; i < 6; i++) {
        compressHDRMip(texture, cubeMap.face(i), i, mipLevel);
    }
}

float evaluateGGX(float roughness, const float cosAngle) {
    float denom;
    roughness *= roughness;
    denom = (roughness - 1)*cosAngle*cosAngle + 1;
    return roughness / (M_PI*denom*denom);
}

float findGGXCosLimitAngle(const float roughness, const float eps) {
    // Do a simple dichotomy search for the moment. We can switch to Newton-Raphson later on or even
    // a closed solution if we can find one...
    float minCosAngle = 0.f;
    float maxCosAngle = 1.f;
    float midCosAngle;
    float x;

    midCosAngle = (maxCosAngle + minCosAngle) / 2.f;
    while ((maxCosAngle - minCosAngle) > 1e-2f) {
        x = evaluateGGX(roughness, midCosAngle) * midCosAngle;
        if (x > eps) {
            maxCosAngle = midCosAngle;
        } else {
            minCosAngle = midCosAngle;
        }
        midCosAngle = (maxCosAngle + minCosAngle) / 2.f;
    }
    return midCosAngle;
}

glm::vec4 applyGGXFilter(const nvtt::CubeSurface& sourceCubeMap, const glm::vec3& filterDir, const float roughness, const float coneCosAngle,
    const image::TexelTable& texelTable) {
    const float coneAngle = acosf(coneCosAngle);
    assert(coneCosAngle >= 0);

    const int size = sourceCubeMap.face(0).width();
    const auto atanSqrt2 = atanf(sqrtf(2));
    glm::vec4 color(0);
    float sum = 0;

    // For each texel of the input cube.
    for (uint f = 0; f < 6; f++) {

        // Test face cone agains filter cone.
        float cosineFaceAngle = glm::dot(filterDir, faceNormals[f]);
        float faceAngle = acosf(cosineFaceAngle);

        if (faceAngle > coneAngle + atanSqrt2) {
            // Skip face.
            continue;
        }

        const int L = int(size - 1);
        int x0 = 0, x1 = L;
        int y0 = 0, y1 = L;

        assert(x1 >= x0);
        assert(y1 >= y0);

        if (x1 == x0 || y1 == y0) {
            // Skip this face.
            continue;
        }


        const nvtt::Surface & inputFace = sourceCubeMap.face(f);
        const float* inputR = inputFace.channel(0);
        const float* inputG = inputFace.channel(1);
        const float* inputB = inputFace.channel(2);
        glm::vec4 pixel(0,0,0,1);

        for (int y = y0; y <= y1; y++) {
            bool inside = false;
            for (int x = x0; x <= x1; x++) {

                glm::vec3 dir = texelTable.direction(f, x, y);
                float cosineAngle = dot(dir, filterDir);

                if (cosineAngle > coneCosAngle) {
                    float solidAngle = texelTable.solidAngle(f, x, y);
                    float scale = evaluateGGX(roughness, cosineAngle);
                    float contribution = solidAngle * scale;
                    int inputIdx = y * size + x;

                    sum += contribution;
                    pixel.r = inputR[inputIdx];
                    pixel.g = inputG[inputIdx];
                    pixel.b = inputB[inputIdx];
                    color += pixel * contribution;

                    inside = true;
                } else if (inside) {
                    // Filter scale is monotonic, if we have been inside once and we just exit, then we can skip the rest of the row.
                    // We could do the same thing for the columns and skip entire rows.
                    break;
                }
            }
        }
    }

    color *= (1.0f / sum);

    return color;
}

void convolveWithGGXLobe(const nvtt::CubeSurface& sourceCubeMap, nvtt::CubeSurface& filteredCubeMap, int faceIndex, 
    const float roughness, const float coneCosAngle, const image::TexelTable& texelTable) {
    nvtt::Surface& filteredFace = filteredCubeMap.face(faceIndex);
    const uint size = filteredFace.width();
    std::vector<glm::vec4> filteredData;
    std::vector<glm::vec4>::iterator filteredDataIt;

    filteredData.resize(size*size);

    filteredDataIt = filteredData.begin();
    for (uint y = 0; y < size; y++) {
        for (uint x = 0; x < size; x++) {
            const glm::vec3 filterDir = texelDirection(faceIndex, x, y, size);
            // Convolve filter against cube.
            glm::vec4 color = applyGGXFilter(sourceCubeMap, filterDir, roughness, coneCosAngle, texelTable);
            *filteredDataIt = color;
            ++filteredDataIt;
        }
    }

    filteredFace.setImage(nvtt::InputFormat_RGBA_32F, size, size, 1, &(*filteredData.begin()));
}

void convolveWithGGXLobe(const nvtt::CubeSurface& sourceCubeMap, nvtt::CubeSurface& destCubeMap, const float roughness,
    const image::TexelTable& texelTable) {
    // This entire code is inspired by the NVTT source code for applying a cosinePowerFilter which is unfortunately private. 
    // If we could give it our proper filter kernel, we woudn't have to do most of this...
    const float threshold = 0.001f;
    // We limit the cone angle of the filter kernel to speed things up
    const float coneCosAngle = findGGXCosLimitAngle(roughness, threshold);

    for (auto i = 0; i < 6; i++) {
        convolveWithGGXLobe(sourceCubeMap, destCubeMap, i, roughness, coneCosAngle, texelTable);
    }
}

float computeGGXRoughnessFromMipLevel(const int mipCount, int mipLevel, float bias) {
    float alpha;

    alpha = glm::clamp(mipCount - mipLevel - bias + 1, 1.f / 12.f, 12.f / 7.f);
    alpha = glm::clamp(2.f / alpha - 1.f / 6.f, 0.f, 1.f);
    return alpha*alpha;
}

namespace image {

    void generateGGXFilteredMips(gpu::Texture* texture, const CubeFaces& faces, gpu::Element sourceFormat) {
        const int size = faces.front().width();
        nvtt::CubeSurface cubeMap;
        nvtt::CubeSurface filteredCubeMap;
        const int mipCount = (int)ceilf(log2f(size));
        const float bias = 0.5f;
        int mipLevel = 0;
        float roughness;
        std::auto_ptr<TexelTable> texelTable( new TexelTable(size) );

        // First pass: convert cube map to vec4 for faster access
        {
            std::vector<glm::vec4>  data;

            for (auto i = 0; i < 6; i++) {
                convertQImageToVec4s(faces[i], sourceFormat, data);
                cubeMap.face(i).setImage(nvtt::InputFormat_RGBA_32F, size, size, 1, &(*data.begin()));
            }
        }

        roughness = computeGGXRoughnessFromMipLevel(mipCount, mipLevel, bias);
        convolveWithGGXLobe(cubeMap, filteredCubeMap, roughness, *texelTable);
        compressHDRCubeMap(texture, filteredCubeMap, mipLevel++);
        while (cubeMap.face(0).canMakeNextMipmap()) {
            assert(mipLevel < mipCount);
            roughness = computeGGXRoughnessFromMipLevel(mipCount, mipLevel, bias);
            convolveWithGGXLobe(cubeMap, filteredCubeMap, roughness, *texelTable);
            for (auto i = 0; i < 6; i++) {
                cubeMap.face(i).buildNextMipmap(nvtt::MipmapFilter_Box);
            }
            texelTable.reset(new TexelTable(cubeMap.face(0).width()));
            compressHDRCubeMap(texture, filteredCubeMap, mipLevel++);
        }
    }
}
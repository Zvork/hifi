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
#include <TBBHelpers.h>

// Necessary for M_PI definition
#include <qmath.h>

#include "ImageLogging.h"

static const glm::vec3 faceNormals[6] = {
    glm::vec3(1, 0, 0),
    glm::vec3(-1, 0, 0),
    glm::vec3(0, 1, 0),
    glm::vec3(0, -1, 0),
    glm::vec3(0, 0, 1),
    glm::vec3(0, 0, -1),
};


static const glm::vec3 faceU[6] = {
    glm::vec3(0, 0, -1),
    glm::vec3(0, 0, 1),
    glm::vec3(1, 0, 0),
    glm::vec3(1, 0, 0),
    glm::vec3(1, 0, 0),
    glm::vec3(-1, 0, 0),
};

static const glm::vec3 faceV[6] = {
    glm::vec3(0, -1, 0),
    glm::vec3(0, -1, 0),
    glm::vec3(0, 0, 1),
    glm::vec3(0, 0, -1),
    glm::vec3(0, -1, 0),
    glm::vec3(0, -1, 0),
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

inline float pixel(const nvtt::Surface& image, uint c, int x, int y) {
    return image.channel(c)[x + y*image.width()];
}

static float bilerp(const nvtt::Surface& image, uint c, int ix0, int iy0, int ix1, int iy1, float fx, float fy) {
    float f1 = pixel(image, c, ix0, iy0);
    float f2 = pixel(image, c, ix1, iy0);
    float f3 = pixel(image, c, ix0, iy1);
    float f4 = pixel(image, c, ix1, iy1);

    float i1 = glm::mix(f1, f2, fx);
    float i2 = glm::mix(f3, f4, fx);

    return glm::mix(i1, i2, fy);
}

static glm::vec4 sampleLinearClamp(const nvtt::Surface& image, float x, float y) {
    const int w = image.width();
    const int h = image.height();

    x = x*w-0.5f;
    y = y*h-0.5f;

    const float fracX = x-floorf(x);
    const float fracY = y-floorf(y);

    const int ix0 = glm::clamp(int(x), 0, w - 1);
    const int iy0 = glm::clamp(int(y), 0, h - 1);
    const int ix1 = glm::clamp(int(x) + 1, 0, w - 1);
    const int iy1 = glm::clamp(int(y) + 1, 0, h - 1);

    glm::vec4 color;
    color.r = bilerp(image, 0, ix0, iy0, ix1, iy1, fracX, fracY);
    color.g = bilerp(image, 1, ix0, iy0, ix1, iy1, fracX, fracY);
    color.b = bilerp(image, 2, ix0, iy0, ix1, iy1, fracX, fracY);
    color.a = 1.f;
    return color;
}

static glm::vec4 sample(const nvtt::CubeSurface& cubeMap, glm::vec3 dir) {
    int f = -1;
    glm::vec3 absDir = glm::abs(dir);

    if (absDir.x > absDir.y && absDir.x > absDir.z) {
        f = dir.x > 0 ? gpu::Texture::CUBE_FACE_RIGHT_POS_X : gpu::Texture::CUBE_FACE_LEFT_NEG_X;
        dir /= absDir.x;
    }
    else if (absDir.y > absDir.z) {
        f = dir.y > 0 ? gpu::Texture::CUBE_FACE_TOP_POS_Y : gpu::Texture::CUBE_FACE_BOTTOM_NEG_Y;
        dir /= absDir.y;
    }
    else {
        f = dir.z > 0 ? gpu::Texture::CUBE_FACE_BACK_POS_Z : gpu::Texture::CUBE_FACE_FRONT_NEG_Z;
        dir /= absDir.z;
    }
    assert(f != -1);

    // uv coordinates corresponding to dir.
    float u = glm::dot(dir, faceU[f])*0.5f + 0.5f;
    float v = glm::dot(dir, faceV[f])*0.5f + 0.5f;

    const nvtt::Surface& img = cubeMap.face(f);

    return sampleLinearClamp(img, u, v);
}

static float evaluateGGX(float roughness, const float cosAngle) {
    if (cosAngle > 0.f) {
        float denom;
        roughness *= roughness;
        denom = (roughness - 1)*cosAngle*cosAngle + 1;
        return roughness / (M_PI*denom*denom);
    } else {
        return 0.f;
    }
}

#define SPECULAR_CONVOLUTION_NORMAL         0
#define SPECULAR_CONVOLUTION_MONTE_CARLO    1

#define SPECULAR_CONVOLUTION_METHOD         SPECULAR_CONVOLUTION_MONTE_CARLO

namespace image {

    void compressHDRMip(gpu::Texture* texture, const nvtt::Surface& surface, int face, int mipLevel);
    void convertQImageToVec4s(const QImage& image, const gpu::Element format, std::vector<glm::vec4>& vec4s);

#if SPECULAR_CONVOLUTION_METHOD==SPECULAR_CONVOLUTION_NORMAL
    struct TexelTable {

        TexelTable(uint edgeLength) : size(edgeLength) {
            // Round up as size isn't necessarily a power of 2
            uint hsize = (size / 2) + (size & 1);
            float hsizeF = size*0.5f;

            // Allocate a small solid angle table that takes into account cube map symmetry.
            solidAngleArray.resize(hsize * hsize);

            for (uint y = 0; y < hsize; y++) {
                for (uint x = 0; x < hsize; x++) {
                    solidAngleArray[y * hsize + x] = solidAngleTerm(hsizeF + x, hsizeF + y, 1.0f / edgeLength);
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
            uint hsize_floor = size / 2;
            uint hsize_ceil = hsize_floor + (size & 1);
            if (x >= hsize_ceil) {
                x -= hsize_floor;
            } else if (x < hsize_ceil) {
                x = hsize_ceil - x - 1;
            }

            if (y >= hsize_ceil) {
                y -= hsize_floor;
            } else if (y < hsize_ceil) {
                y = hsize_ceil - y - 1;
            }

            return solidAngleArray[y * hsize_ceil + x];
        }

        const glm::vec3 & direction(uint f, uint x, uint y) const {
            assert(f < 6 && x < size && y < size);
            return directionArray[(f * size + y) * size + x];
        }

        uint size;
        std::vector<float> solidAngleArray;
        std::vector<glm::vec3> directionArray;
    };
#endif
}

static void compressHDRCubeMap(gpu::Texture* texture, const nvtt::CubeSurface& cubeMap, int mipLevel) {
    for (auto i = 0; i < 6; i++) {
        image::compressHDRMip(texture, cubeMap.face(i), i, mipLevel);
    }
}

#if SPECULAR_CONVOLUTION_METHOD==SPECULAR_CONVOLUTION_NORMAL

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

static float findSpecularCosLimitAngle(const float roughness, const float eps) {
    // Do a simple dichotomy search for the moment. We can switch to Newton-Raphson later on or even
    // a closed solution if we can find one...
    float minCosAngle = 0.f;
    float maxCosAngle = 1.f;
    float midCosAngle;
    float x;

    midCosAngle = (maxCosAngle + minCosAngle) / 2.f;
    while ((maxCosAngle - minCosAngle) > 1e-3f) {
        x = evaluateGGX(roughness, midCosAngle) * midCosAngle;
        if (x > eps) {
            maxCosAngle = midCosAngle;
        }
        else {
            minCosAngle = midCosAngle;
        }
        midCosAngle = (maxCosAngle + minCosAngle) / 2.f;
    }
    return midCosAngle;
}

struct ConvolutionConfig
{
    float coneCosAngle;
    const image::TexelTable& texelTable;

    ConvolutionConfig(float roughness, const image::TexelTable& texelTable) : texelTable(texelTable) {
        // This entire code is inspired by the NVTT source code for applying a cosinePowerFilter which is unfortunately private. 
        // If we could give it our proper filter kernel, we woudn't have to do most of this...
        const float threshold = 0.001f;
        // We limit the cone angle of the filter kernel to speed things up
        coneCosAngle = findSpecularCosLimitAngle(roughness, threshold);
    }
};

static glm::vec4 applySpecularFilter(const nvtt::CubeSurface& sourceCubeMap, const glm::vec3& filterDir, const float roughness, const ConvolutionConfig& config) {
    const float coneAngle = acosf(config.coneCosAngle);
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

        if (size>1 && (x1 == x0 || y1 == y0)) {
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

                glm::vec3 dir = config.texelTable.direction(f, x, y);
                float cosineAngle = dot(dir, filterDir);

                if (cosineAngle > config.coneCosAngle) {
                    float solidAngle = config.texelTable.solidAngle(f, x, y);
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
#elif SPECULAR_CONVOLUTION_METHOD==SPECULAR_CONVOLUTION_MONTE_CARLO
class ConvolutionConfig
{
public:
   
    ConvolutionConfig(uint sampleCount, const nvtt::CubeSurface& cubeMap) :
        _sampleCount(sampleCount)
    {
        assert(_sampleCount > 0);
        // Create a lat / long importance map for importance sampling of the cubemap.
        _cdfSize.x = cubeMap.face(0).width() * 6; // We multiply by 6 to slightly oversample the cube map
        _cdfSize.y = _cdfSize.x / 2;
        _cdfX.resize(_cdfSize.x*_cdfSize.y);
        _cdfY.resize(_cdfSize.y);
        initializeImportanceMap(cubeMap);
    }

    uint sampleCount() const {
        return _sampleCount;
    }
    
    glm::vec3 getCubeMapImportanceSampledDir(const glm::vec2& random, float& probability) const {
        // Start by choosing the row index
        auto rowIt = std::lower_bound(_cdfY.begin(), _cdfY.end(), random.y);
        assert(rowIt != _cdfY.end());
        auto rowIndex = std::distance(_cdfY.begin(), rowIt);
        auto rowOffset = rowIndex * _cdfSize.x;
        auto columnBegin = _cdfX.begin() + rowOffset;
        auto columnIt = std::lower_bound(columnBegin, columnBegin + _cdfSize.x, random.x);
        auto columnIndex = std::distance(columnBegin, columnIt);

        const float elevation = (rowIndex * M_PI) / (_cdfSize.y - 1);
        const float sinElevation = sinf(elevation);
        const float cosElevation = cosf(elevation);
        const float azimuth = (columnIndex * 2 * M_PI) / _cdfSize.x;
        const float sinAzimuth = sinf(azimuth);
        const float cosAzimuth = cosf(azimuth);
        glm::vec3 dir;

        dir.x = sinAzimuth * sinElevation;
        dir.y = cosElevation;
        dir.z = cosAzimuth * sinElevation;
        // Compute the probability that this sample dir was generated
        float probabilityY = *rowIt;
        if (rowIt != _cdfY.begin()) {
            --rowIt;
            probabilityY -= *rowIt;
        }
        float probabilityX = *columnIt;
        if (columnIt != columnBegin) {
            --columnIt;
            probabilityX -= *columnIt;
        }
        probability = probabilityX * probabilityY;

        return dir;
    }

    float getProbabilityOfDir(const glm::vec3& dir) const {
        int y = std::min<int>(acosf(dir.y) * (_cdfSize.y - 1) / M_PI, _cdfSize.y - 1);
        int x = int((atan2f(dir.x, dir.z)+M_PI) * _cdfSize.x / (2*M_PI)) % _cdfSize.x;
        float probabilityX;
        float probabilityY;

        probabilityY = _cdfY[y];
        if (y > 0) {
            probabilityY -= _cdfY[y - 1];
        }
        y *= _cdfSize.x;
        probabilityX = _cdfX[x+y];
        if (x > 0) {
            probabilityX -= _cdfX[x-1 + y];
        }
        return probabilityX*probabilityY;
    }

private:

    uint _sampleCount;
    std::vector<float> _cdfX;
    std::vector<float> _cdfY;
    glm::ivec2 _cdfSize;

    // We create a cumulative distribution function by integrating probabilities in X and Y
    // from the luminance of the cube map.
    // Each row of CDF X contains the integral of the probabilities from left to right of pixels
    // of fixed elevation.
    // Each CDF Y contains the integral of the environment map row probabilities
    void initializeImportanceMap(const nvtt::CubeSurface& cubeMap) {
        glm::vec3 dir;
        glm::vec4 color;
        float probability;
        float probabilityXIntegral;
        float probabilityYIntegral;
        std::vector<float>::iterator importanceMapIt;
        std::vector<float>::iterator importanceMapLineBegin;
        std::vector<float>::iterator importanceMapLineIt;
        int x, y;

        importanceMapIt = _cdfX.begin();
        probabilityYIntegral = 0.f;
        for (y = 0; y < _cdfSize.y; y++) {
            const float elevation = (y * M_PI) / (_cdfSize.y - 1);
            const float sinElevation = sinf(elevation);
            const float cosElevation = cosf(elevation);

            importanceMapLineBegin = importanceMapIt;
            probabilityXIntegral = 0.f;
            for (x = 0; x < _cdfSize.x; x++) {
                const float azimuth = (x * 2 * M_PI) / _cdfSize.x;
                const float sinAzimuth = sinf(azimuth);
                const float cosAzimuth = cosf(azimuth);

                dir.x = sinAzimuth * sinElevation;
                dir.y = cosElevation;
                dir.z = cosAzimuth * sinElevation;

                // Start by sampling the cubeMap's weighted luminance values to
                // compute the probability of each direction
                color = sample(cubeMap, dir);
                probability = (color.r + color.g + color.b) * sinElevation;

                // Integrate it in the x direction
                probabilityXIntegral += probability;
                *importanceMapIt = probabilityXIntegral;
                ++importanceMapIt;
            }

            // Normalize the probabilities in the x direction
            if (probabilityXIntegral > 0.f) {
                for (importanceMapLineIt = importanceMapLineBegin; importanceMapLineIt != importanceMapIt; ++importanceMapLineIt) {
                    *importanceMapLineIt /= probabilityXIntegral;
                }
            }

            // This is the non normalized probability for this row
            probabilityYIntegral += probabilityXIntegral;
            _cdfY[y] = probabilityYIntegral;
        }

        // Normalize the probabilities in the y direction
        if (probabilityYIntegral > 0) {
            for (importanceMapLineIt = _cdfY.begin(); importanceMapLineIt != _cdfY.end(); ++importanceMapLineIt) {
                *importanceMapLineIt /= probabilityYIntegral;
            }
        }
    }
};

// Code taken from https://learnopengl.com/#!PBR/IBL/Specular-IBL
static float getRadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

inline glm::vec2 generateHammersley(uint i, uint N) {
    return glm::vec2(float(i) / float(N), getRadicalInverse_VdC(i));
}

static glm::vec3 getGGXImportanceSampledDir(glm::vec2 Xi, glm::vec3 N, float roughness, float& probability)
{
    float a = roughness;

    float phi = 2.0 * M_PI * Xi.x;
    float cosTheta = sqrtf((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrtf(1.0 - cosTheta*cosTheta);

    // from spherical coordinates to cartesian coordinates
    glm::vec3 H;
    H.x = cosf(phi) * sinTheta;
    H.y = sinf(phi) * sinTheta;
    H.z = cosTheta;

    // from tangent-space vector to world-space sample vector
    glm::vec3 up = fabsf(N.z) < 0.999 ? glm::vec3(0.0, 0.0, 1.0) : glm::vec3(1.0, 0.0, 0.0);
    glm::vec3 tangent = glm::normalize(glm::cross(up, N));
    glm::vec3 bitangent = glm::cross(N, tangent);

    glm::vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;

    // Compute the probability that this direction was generated
    probability = evaluateGGX(roughness, cosTheta);

    return glm::normalize(sampleVec);
}

static glm::vec4 sampleBRDF(const glm::vec2& randomSeed, const nvtt::CubeSurface& sourceCubeMap, const glm::vec3& filterDir,
    const float roughness, const ConvolutionConfig& config) {
    float probabilityGGX;
    float probabilityCubeMap;
    float weight = 1.f;
    glm::vec3 viewDir = filterDir;
    glm::vec3 halfDir = getGGXImportanceSampledDir(randomSeed, filterDir, roughness, probabilityGGX);
    glm::vec3 lightDir = glm::normalize(2.0f * glm::dot(viewDir, halfDir) * halfDir - viewDir);
    glm::vec4 color(0, 0, 0, 1);
    float NdotL = glm::dot(filterDir, lightDir);

    if (NdotL > 0.f) {
        probabilityCubeMap = config.getProbabilityOfDir(lightDir);
        // Combine the two for multiple importance sampling based on the balance heuristic
        weight = probabilityGGX + probabilityCubeMap;
        if (weight > 0.f) {
            // The probabilityGGX is the GGX NDF
            color = sample(sourceCubeMap, lightDir) * NdotL * probabilityGGX;
            color /= weight;
        }
    }
    return color;
}

static glm::vec4 sampleCubeMap(const glm::vec2& randomSeed, const nvtt::CubeSurface& sourceCubeMap, const glm::vec3& filterDir,
    const float roughness, const ConvolutionConfig& config) {
    float probabilityGGX;
    float probabilityCubeMap;
    float weight = 1.f;
    glm::vec3 viewDir = filterDir;
    glm::vec3 lightDir = config.getCubeMapImportanceSampledDir(randomSeed, probabilityCubeMap);
    glm::vec4 color(0, 0, 0, 1);
    float NdotL;

    NdotL = glm::dot(filterDir, lightDir);
    if (NdotL > 0.f) {
        glm::vec3 halfDir = glm::normalize(lightDir + viewDir);

        probabilityGGX = evaluateGGX(roughness, glm::dot(halfDir, filterDir));
        // Combine the two for multiple importance sampling based on the balance heuristic
        weight = probabilityGGX + probabilityCubeMap;
        if (weight > 0.f) {
            // The probabilityGGX is the GGX NDF
            color = sample(sourceCubeMap, lightDir) * NdotL * probabilityGGX;
            color /= weight;
        }
    }
    return color;
}

static glm::vec4 applySpecularFilter(const nvtt::CubeSurface& sourceCubeMap, const glm::vec3& filterDir, const float roughness, const ConvolutionConfig& config,
    std::vector<glm::vec2>::const_iterator randomSeedIt) {
    glm::vec4 filteredColor{ 0,0,0,1 };
    glm::vec2 randomSeed;

    for (uint i = 0; i < config.sampleCount(); i++) {
        randomSeed = *randomSeedIt;
        ++randomSeedIt;

        // First generate a sample based on the GGX distribution
        filteredColor += sampleBRDF(randomSeed, sourceCubeMap, filterDir, roughness, config);
  
        // Then another sample based on the cubemap distribution
        filteredColor += sampleCubeMap(randomSeed, sourceCubeMap, filterDir, roughness, config);
    }
    // Both sampling strategies BSDF & EnvMap have the same number of samples
    filteredColor /= config.sampleCount();
    return filteredColor;
}

#endif

static void generateHammersleySequence(std::vector<glm::vec2>& sequence) {
    int i = 1;

    for(auto& value : sequence) {
        value = generateHammersley(i++, sequence.size()+1);
    }
}

static void convolveWithSpecularLobe(const nvtt::CubeSurface& sourceCubeMap, nvtt::CubeSurface& filteredCubeMap, int faceIndex,
    const float roughness, const ConvolutionConfig& config) {
    nvtt::Surface& filteredFace = filteredCubeMap.face(faceIndex);
    const uint size = sourceCubeMap.face(0).width();
    std::vector<glm::vec4> filteredData;
    std::vector<glm::vec2> randomSeeds;

    randomSeeds.resize(config.sampleCount());
    generateHammersleySequence(randomSeeds);

    filteredData.resize(size*size);

#if 1
    std::vector<glm::vec4>::iterator filteredDataIt = filteredData.begin();

    for (uint y = 0; y < size; y++) {
        for (uint x = 0; x < size; x++) {
            const glm::vec3 filterDir = texelDirection(faceIndex, x, y, size);
            // Convolve filter against cube.
            glm::vec4 color = applySpecularFilter(sourceCubeMap, filterDir, roughness, config, randomSeeds.begin());
            *filteredDataIt = color;
            ++filteredDataIt;
        }
    }
#else
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size*size), [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); i++) {
            int x = int(i % size);
            int y = int(i / size);
            const glm::vec3 filterDir = texelDirection(faceIndex, x, y, size);
            // Convolve filter against cube.
            glm::vec4 color = applySpecularFilter(sourceCubeMap, filterDir, roughness, config, randomSeeds.begin());
            filteredData[x + y*size] = color;
        }
    });
#endif

    filteredFace.setImage(nvtt::InputFormat_RGBA_32F, size, size, 1, &(*filteredData.begin()));
}

static void convolveWithSpecularLobe(const nvtt::CubeSurface& sourceCubeMap, nvtt::CubeSurface& destCubeMap, const float roughness,
    const ConvolutionConfig& config) {
    for (auto i = 0; i < 6; i++) {
        convolveWithSpecularLobe(sourceCubeMap, destCubeMap, i, roughness, config);
    }
}

static float computeGGXRoughnessFromMipLevel(const int size, int mipLevel, float bias) {
    float alpha;
    float mipCount = ceilf(log2f(size));

    alpha = glm::clamp(mipCount - mipLevel - bias, 26 / 15.f, 13.f);
    alpha = glm::clamp(2.f / alpha - 2.f / 13.f, 0.f, 1.f);
    return alpha*alpha;
}

namespace image {

    void generateSpecularFilteredMips(gpu::Texture* texture, const CubeFaces& faces, gpu::Element sourceFormat, const std::string& srcImageName) {
        const int size = faces.front().width();
        nvtt::CubeSurface cubeMap;
        nvtt::CubeSurface filteredCubeMap;
        const float bias = 0.5f;
        int mipLevel = 0;
        float roughness;
#if SPECULAR_CONVOLUTION_METHOD==SPECULAR_CONVOLUTION_NORMAL
        std::auto_ptr<TexelTable> texelTable( new TexelTable(size) );
#endif

        // First pass: convert cube map to vec4 for faster access
        {
            std::vector<glm::vec4>  data;

            for (auto i = 0; i < 6; i++) {
                image::convertQImageToVec4s(faces[i], sourceFormat, data);
                cubeMap.face(i).setImage(nvtt::InputFormat_RGBA_32F, size, size, 1, &(*data.begin()));
            }
        }

        // This is a compromise between speed and precision: building the mip maps
        // on the source cube map and then applying the GGX convolution results in extra
        // filtering due to the box filtering used in the mip building function.
        // In theory, to prevent this we should compute the filtered results at
        // full resolution for each mip and then downsize each filtered result to the final 
        // mip resolution without any extra filtering. This would work as the GGX filters
        // act as low pass filters.

#if SPECULAR_CONVOLUTION_METHOD == SPECULAR_CONVOLUTION_MONTE_CARLO
            ConvolutionConfig config( 64U, cubeMap );
#endif
        // First level is always RAW
//        roughness = computeGGXRoughnessFromMipLevel(size, mipLevel, bias);
//        convolveWithSpecularLobe(cubeMap, filteredCubeMap, roughness, *texelTable);
        compressHDRCubeMap(texture, cubeMap, mipLevel++);
        qCInfo(imagelogging) << "Cube map " << QString(srcImageName.c_str()) << " mip level " << (mipLevel-1) << '/' << texture->getMaxMip() << " has been processed.";
        while (cubeMap.face(0).canMakeNextMipmap()) {
            roughness = computeGGXRoughnessFromMipLevel(size, mipLevel, bias);
#if SPECULAR_CONVOLUTION_METHOD==SPECULAR_CONVOLUTION_NORMAL
            texelTable.reset(new TexelTable(cubeMap.face(0).width()));
            ConvolutionConfig config{ roughness, *texelTable };
#endif
            convolveWithSpecularLobe(cubeMap, filteredCubeMap, roughness, config);
            for (auto i = 0; i < 6; i++) {
                cubeMap.face(i).buildNextMipmap(nvtt::MipmapFilter_Box);
                filteredCubeMap.face(i).buildNextMipmap(nvtt::MipmapFilter_Box);
            }
            compressHDRCubeMap(texture, filteredCubeMap, mipLevel++);
            qCInfo(imagelogging) << "Cube map " << QString(srcImageName.c_str()) << " mip level " << (mipLevel - 1) << " has been processed.";
        }
    }
}
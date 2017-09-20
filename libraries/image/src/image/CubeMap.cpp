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

#include <chrono>

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

static glm::vec3 getGGXImportanceSampledHalfDir(glm::vec2 Xi, float roughness, float& pdf) {
    float a = roughness;

    float phi = 2.0 * M_PI * Xi.x;
    float cosTheta = sqrtf((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrtf(1.0 - cosTheta*cosTheta);

    // from spherical coordinates to cartesian coordinates
    glm::vec3 H;
    H.x = cosf(phi) * sinTheta;
    H.y = sinf(phi) * sinTheta;
    H.z = cosTheta;

    // Compute the pdf that this direction was generated
    pdf = evaluateGGX(roughness, cosTheta);

    return H;
}

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
   
    ConvolutionConfig(uint sampleCountBRDF, uint sampleCountEnv, const nvtt::CubeSurface& cubeMap) :
        _sampleCountBRDF(sampleCountBRDF), _sampleCountEnv(sampleCountEnv) {
        glm::vec3 dir;
        glm::vec4 color;
        float pdf;
        float cdfX;
        float cdfY;
        FloatVector cdfXArray;
        FloatVector cdfYArray;
        FloatVector::iterator cdfIterator;
        FloatVector::iterator cdfLineBegin;
        FloatVector::iterator cdfLineIterator;
        FloatVector::iterator pdfIterator;
        int x, y;

        // We create a cumulative distribution function by integrating probability densities in X and Y
        // from the luminance of the cube map.
        // Each row of CDF X contains the integral of the probability density from left to right of pixels
        // of fixed elevation.
        // Each CDF Y contains the integral of the environment map row probability density

        assert((_sampleCountBRDF+ _sampleCountEnv) > 0);
        // Create a lat / long importance map for importance sampling of the cubemap.
        _cdfSize.x = cubeMap.face(0).width() * 6; // We multiply by 6 to slightly oversample the cube map
        _cdfSize.y = _cdfSize.x / 2;
        cdfXArray.resize(_cdfSize.x*_cdfSize.y);
        cdfYArray.resize(_cdfSize.y);
        _inverseCDFX.resize(_cdfSize.x*_cdfSize.y);
        _inverseCDFY.resize(_cdfSize.y);
        _pdf.resize(_cdfSize.x*_cdfSize.y);

        cdfIterator = cdfXArray.begin();
        pdfIterator = _pdf.begin();
        cdfY = 0.f;
        for (y = 0; y < _cdfSize.y; y++) {
            const float elevation = (y * M_PI) / (_cdfSize.y - 1);
            const float sinElevation = sinf(elevation);
            const float cosElevation = cosf(elevation);

            cdfLineBegin = cdfIterator;
            cdfX = 0.f;
            for (x = 0; x < _cdfSize.x; x++) {
                const float azimuth = (x * 2 * M_PI) / _cdfSize.x;
                const float sinAzimuth = sinf(azimuth);
                const float cosAzimuth = cosf(azimuth);

                dir.x = sinAzimuth * sinElevation;
                dir.y = cosElevation;
                dir.z = cosAzimuth * sinElevation;

                // Start by sampling the cubeMap's weighted luminance values to
                // compute the probability density of each direction. We weight it
                // by the sine of the elevation because the solid angle of that pixel
                // becomes smaller with the elevation.
                color = sample(cubeMap, dir);
                pdf = (color.r + color.g + color.b) * sinElevation;
                *pdfIterator = pdf;
                ++pdfIterator;

                // Integrate it in the x direction
                cdfX += pdf;
                *cdfIterator = cdfX;
                ++cdfIterator;
            }

            // Normalize the CDF in the x direction
            if (cdfX > 0.f) {
                for (cdfLineIterator = cdfLineBegin; cdfLineIterator != cdfIterator; ++cdfLineIterator) {
                    *cdfLineIterator /= cdfX;
                }
            }

            // This is the non normalized CDF for this row
            cdfY += cdfX;
            cdfYArray[y] = cdfY;
        }

        // Normalize the PDF by dividing by the total sum of all weighted luminances which happens to be the last element
        // of cdfY as it hasn't been normalized yet. But we also need to multiply by the total number of pixels in the CDF
        // because this is the inverse of the sampling rate and our PDF is a density, thus a derivative
        const float normalizer = _cdfSize.x * _cdfSize.y / cdfY;
        for (auto& pdf : _pdf) {
            pdf *= normalizer;
        }

        // Normalize the CDF in the y direction
        if (cdfY > 0) {
            for (cdfLineIterator = cdfYArray.begin(); cdfLineIterator != cdfYArray.end(); ++cdfLineIterator) {
                *cdfLineIterator /= cdfY;
            }
        }

        // Final step: create the invert of both cdf functions for faster lookup.
        invertCDF(cdfYArray.begin(), cdfYArray.end(), _inverseCDFY.begin(), _inverseCDFY.end());
        for (y = 0; y < _cdfSize.y; y++) {
            auto offset = y*_cdfSize.x;
            cdfLineBegin = cdfXArray.begin() + y*_cdfSize.x;
            invertCDF(cdfLineBegin, cdfLineBegin + _cdfSize.x, _inverseCDFX.begin() + offset, _inverseCDFX.begin() + offset + _cdfSize.x);
        }
    }

    inline uint sampleCountForBRDF() const {
        return _sampleCountBRDF;
    }

    inline uint sampleCountForEnvironment() const {
        return _sampleCountEnv;
    }

    glm::vec3 getCubeMapImportanceSampledDir(const glm::vec2& random, float& pdf) const {
        auto rowIndex = (int)floorf(random.y * (_cdfSize.y-1) + 0.5f);
        auto columnIndex = (int)floorf(random.x * (_cdfSize.x-1) + 0.5f);

        rowIndex = _inverseCDFY[rowIndex];
        columnIndex = _inverseCDFX[columnIndex + rowIndex*_cdfSize.x];

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

        pdf = getProbabilityDensity(columnIndex, rowIndex);

        return dir;
    }

    float getProbabilityDensityOfDir(const glm::vec3& dir) const {
        int y = std::min<int>(acosf(dir.y) * (_cdfSize.y - 1) / M_PI, _cdfSize.y - 1);
        float azimuth = atan2f(dir.x, dir.z) + M_PI;
        assert(azimuth >= 0.f && azimuth <= 2 * M_PI);
        int x = int(azimuth * _cdfSize.x / (2*M_PI)) % _cdfSize.x;
        assert(y >= 0);
        return getProbabilityDensity(x, y);
    }

private:

    typedef std::vector<uint>  UIntVector;
    typedef std::vector<float> FloatVector;

    uint _sampleCountBRDF;
    uint _sampleCountEnv;
    UIntVector _inverseCDFX;
    UIntVector _inverseCDFY;
    FloatVector _pdf;
    glm::ivec2 _cdfSize;

    inline float getProbabilityDensity(int x, int y) const {
        return _pdf[x + y*_cdfSize.x];
    }

    void invertCDF(FloatVector::const_iterator beginCDF, FloatVector::const_iterator endCDF, 
                   UIntVector::iterator beginInverseCDF, UIntVector::iterator endInverseCDF) {
        size_t size = std::distance(beginInverseCDF, endInverseCDF);
        double probability = 0.0;
        double deltaProbability = 1.0 / (size -1);

        for (UIntVector::iterator invertIt = beginInverseCDF; invertIt != endInverseCDF; ++invertIt) {
            auto cdfIt = std::lower_bound(beginCDF, endCDF, (float)probability);
            auto index = std::distance(beginCDF, cdfIt);
            assert(cdfIt != endCDF);
            *invertIt = index;
            probability = std::min(1.0, probability+deltaProbability);
        }
    }
};

class SampleSource
{
public:

    SampleSource(const ConvolutionConfig& config, float roughness, int count) {
        _BRDFValues.resize(count);
        _EnvValues.resize(count);

        for (auto i = 0; i < count; i++) {
            glm::vec2 random = generateHammersley(i + 1, count + 1);
            glm::vec3 dir;
            float pdf;

            dir = getGGXImportanceSampledHalfDir(random, roughness, pdf);
            _BRDFValues[i] = glm::vec4(dir, pdf);

            dir = config.getCubeMapImportanceSampledDir(random, pdf);
            _EnvValues[i] = glm::vec4(dir, pdf);
        }
        _nextBRDFValue = _BRDFValues.begin();
        _nextEnvValue = _EnvValues.begin();
    }

    glm::vec4 getBRDF() {
        glm::vec4 value = *_nextBRDFValue;
        ++_nextBRDFValue;
        if (_nextBRDFValue == _BRDFValues.end()) {
            _nextBRDFValue = _BRDFValues.begin();
        }
        return value;
    }

    glm::vec4 getEnvironment() {
        glm::vec4 value = *_nextEnvValue;
        ++_nextEnvValue;
        if (_nextEnvValue == _EnvValues.end()) {
            _nextEnvValue = _EnvValues.begin();
        }
        return value;
    }

private:

    std::vector<glm::vec4>  _BRDFValues;
    std::vector<glm::vec4>  _EnvValues;
    std::vector<glm::vec4>::const_iterator _nextBRDFValue;
    std::vector<glm::vec4>::const_iterator _nextEnvValue;
};

static glm::vec3 toWorldSpace(const glm::vec3& halfDirTangentSpace, glm::vec3 N) {
    // from tangent-space vector to world-space sample vector
    glm::vec3 up = fabsf(N.z) < 0.999 ? glm::vec3(0.0, 0.0, 1.0) : glm::vec3(1.0, 0.0, 0.0);
    glm::vec3 tangent = glm::normalize(glm::cross(up, N));
    glm::vec3 bitangent = glm::cross(N, tangent);

    glm::vec3 sampleVec = tangent * halfDirTangentSpace.x + bitangent * halfDirTangentSpace.y + N * halfDirTangentSpace.z;

    return glm::normalize(sampleVec);
}

static glm::vec4 sampleBRDF(const glm::vec4& randomSample, const nvtt::CubeSurface& sourceCubeMap, const glm::vec3& filterDir,
    const float roughness, const ConvolutionConfig& config) {
    float pdfGGX = randomSample.w;
    float pdfCubeMap;
    glm::vec3 viewDir = filterDir;
    glm::vec3 halfDir = toWorldSpace(glm::vec3(randomSample.x, randomSample.y, randomSample.z), filterDir);
    glm::vec3 lightDir = glm::normalize(2.0f * glm::dot(viewDir, halfDir) * halfDir - viewDir);
    glm::vec4 color(0, 0, 0, 0);
    float NdotL = glm::dot(filterDir, lightDir);

    if (NdotL > 0.f) {
        const double ggxWeight = config.sampleCountForBRDF() * config.sampleCountForBRDF();
        const double cubeMapWeight = config.sampleCountForEnvironment() * config.sampleCountForEnvironment();
        double weight;

        pdfCubeMap = config.getProbabilityDensityOfDir(lightDir);
        // Combine the two for multiple importance sampling based on the power heuristic
        weight = pdfGGX*pdfGGX*ggxWeight + pdfCubeMap*pdfCubeMap*cubeMapWeight;
        if (weight > 0.0) {
            // The pdfGGX is the GGX NDF
            color = sample(sourceCubeMap, lightDir) * pdfGGX * NdotL;
            color *= float(pdfGGX*ggxWeight / weight);
            color.a = NdotL;
        }
    }
    return color;
}

static glm::vec4 sampleCubeMap(const glm::vec4& randomSample, const nvtt::CubeSurface& sourceCubeMap, const glm::vec3& filterDir,
    const float roughness, const ConvolutionConfig& config) {
    float pdfGGX;
    float pdfCubeMap = randomSample.w;
    glm::vec3 viewDir = filterDir;
    glm::vec3 lightDir(randomSample.x, randomSample.y, randomSample.z);
    glm::vec4 color(0, 0, 0, 0);
    float NdotL;

    NdotL = glm::dot(filterDir, lightDir);
    if (NdotL > 0.f) {
        const double ggxWeight = config.sampleCountForBRDF() * config.sampleCountForBRDF();
        const double cubeMapWeight = config.sampleCountForEnvironment() * config.sampleCountForEnvironment();
        double weight;
        glm::vec3 halfDir = glm::normalize(lightDir + viewDir);

        pdfGGX = evaluateGGX(roughness, glm::dot(halfDir, filterDir));
        // Combine the two for multiple importance sampling based on the power heuristic
        weight = pdfGGX*pdfGGX*ggxWeight + pdfCubeMap*pdfCubeMap*cubeMapWeight;
        if (weight > 0.0) {
            // pdfGGX is the GGX NDF
            color = sample(sourceCubeMap, lightDir) * pdfGGX * NdotL;
            color *= float(pdfCubeMap*cubeMapWeight / weight);
            color.a = NdotL;
        }
    }
    return color;
}

static glm::vec4 applySpecularFilter(const nvtt::CubeSurface& sourceCubeMap, const glm::vec3& filterDir, const float roughness, const ConvolutionConfig& config,
    SampleSource& samples) {
    glm::vec4 filteredColor1{ 0,0,0,0 };
    glm::vec4 filteredColor2{ 0,0,0,0 };
    uint i;

    // First generate samples based on the GGX distribution
    for (i = 0; i < config.sampleCountForBRDF(); i++) {
        filteredColor1 += sampleBRDF(samples.getBRDF(), sourceCubeMap, filterDir, roughness, config);
    }
    // The alpha channel stores the sum of NdotLs and we divide the result by this
    // to normalize the total GGX * NdotL PDF to 1.
    if (filteredColor1.a > 0.f) {
        filteredColor1 /= filteredColor1.a;
    }

    // Then other samples based on the cubemap distribution
    for ( i = 0; i < config.sampleCountForEnvironment(); i++) {
        filteredColor2 += sampleCubeMap(samples.getEnvironment(), sourceCubeMap, filterDir, roughness, config);
    }
    // The alpha channel stores the sum of NdotLs and we divide the result by this
    // to normalize the total GGX * NdotL PDF to 1.
    if (filteredColor2.a > 0.f) {
        filteredColor2 /= filteredColor2.a;
    }

    return filteredColor1+filteredColor2;
}

#endif

static void convolveWithSpecularLobe(const nvtt::CubeSurface& sourceCubeMap, nvtt::CubeSurface& filteredCubeMap, int faceIndex,
    const float roughness, const ConvolutionConfig& config) {
    nvtt::Surface& filteredFace = filteredCubeMap.face(faceIndex);
    const uint size = sourceCubeMap.face(0).width();
    std::vector<glm::vec4> filteredData;

    filteredData.resize(size*size);

#if 0
    // Sequential version for debugging
    std::vector<glm::vec4>::iterator filteredDataIt = filteredData.begin();
    SampleSource samples(config, roughness, std::max(config.sampleCountForBRDF(), config.sampleCountForEnvironment()));

    for (uint y = 0; y < size; y++) {
        for (uint x = 0; x < size; x++) {
            const glm::vec3 filterDir = texelDirection(faceIndex, x, y, size);
            // Convolve filter against cube.
            glm::vec4 color = applySpecularFilter(sourceCubeMap, filterDir, roughness, config, samples);
            *filteredDataIt = color;
            ++filteredDataIt;
        }
    }
#else
    // Parallel version for performance
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size*size), [&](const tbb::blocked_range<size_t>& range) {
        SampleSource samples(config, roughness, std::max(config.sampleCountForBRDF(), config.sampleCountForEnvironment()));

        for (size_t i = range.begin(); i != range.end(); i++) {
            int x = int(i % size);
            int y = int(i / size);
            const glm::vec3 filterDir = texelDirection(faceIndex, x, y, size);
            // Convolve filter against cube.
            glm::vec4 color = applySpecularFilter(sourceCubeMap, filterDir, roughness, config, samples);
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
        const float bias = 1.0f;
        int mipLevel = 0;
        float roughness;
#if SPECULAR_CONVOLUTION_METHOD==SPECULAR_CONVOLUTION_NORMAL
        std::auto_ptr<TexelTable> texelTable( new TexelTable(size) );
#endif
        std::chrono::steady_clock::time_point   start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point   end;

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
        ConvolutionConfig config( 150U, 75U, cubeMap );
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

        end = std::chrono::steady_clock::now();
        qCInfo(imagelogging) << "Cube map " << QString(srcImageName.c_str()) << " processed in " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << " seconds.";
    }
}
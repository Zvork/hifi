//
//  CubeMap.h
//  image/src/image
//
//  Created by Olivier Prat on 09/14/2017.
//  Copyright 2017 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#ifndef IMAGE_CUBEMAP_H
#define IMAGE_CUBEMAP_H

#include <QImage>

#include <gpu/Texture.h>

#include <vector>

namespace image {

    typedef std::vector<QImage> CubeFaces;

    void generateGGXFilteredMips(gpu::Texture* texture, const CubeFaces& faces, gpu::Element sourceFormat);

}

#endif

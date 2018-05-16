//
//  RandomAndNoise.h
//
//  Created by Olivier Prat on 05/16/18.
//  Copyright 2018 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#ifndef RANDOM_AND_NOISE_H
#define RANDOM_AND_NOISE_H

// Low discrepancy Halton sequence generator
template <int B>
float evaluateHalton(int index) {
    float f = 1.0f;
    float r = 0.0f;
    float invB = 1.0f / (float)B;
    index++; // Indices start at 1, not 0

    while (index > 0) {
        f = f * invB;
        r = r + f * (float)(index % B);
        index = index / B;

    }

    return r;
}

#endif
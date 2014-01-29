//
//  Particle.cpp
//  hifi
//
//  Created by Brad Hefta-Gaub on 12/4/13.
//  Copyright (c) 2013 High Fidelity, Inc. All rights reserved.
//
//

#include <QtCore/QObject>

#include <Octree.h>
#include <RegisteredMetaTypes.h>
#include <SharedUtil.h> // usecTimestampNow()
#include <VoxelsScriptingInterface.h>

// This is not ideal, but adding script-engine as a linked library, will cause a circular reference
// I'm open to other potential solutions. Could we change cmake to allow libraries to reference each others
// headers, but not link to each other, this is essentially what this construct is doing, but would be
// better to add includes to the include path, but not link
#include "../../script-engine/src/ScriptEngine.h"

#include "ParticlesScriptingInterface.h"
#include "Particle.h"
#include "ParticleTree.h"

uint32_t Particle::_nextID = 0;
VoxelEditPacketSender* Particle::_voxelEditSender = NULL;
ParticleEditPacketSender* Particle::_particleEditSender = NULL;

// for locally created particles
std::map<uint32_t,uint32_t> Particle::_tokenIDsToIDs;
uint32_t Particle::_nextCreatorTokenID = 0;

uint32_t Particle::getIDfromCreatorTokenID(uint32_t creatorTokenID) {
    if (_tokenIDsToIDs.find(creatorTokenID) != _tokenIDsToIDs.end()) {
        return _tokenIDsToIDs[creatorTokenID];
    }
    return UNKNOWN_PARTICLE_ID;
}

uint32_t Particle::getNextCreatorTokenID() {
    uint32_t creatorTokenID = _nextCreatorTokenID;
    _nextCreatorTokenID++;
    return creatorTokenID;
}

void Particle::handleAddParticleResponse(const QByteArray& packet) {
    const unsigned char* dataAt = reinterpret_cast<const unsigned char*>(packet.data());
    int numBytesPacketHeader = numBytesForPacketHeader(packet);
    dataAt += numBytesPacketHeader;

    uint32_t creatorTokenID;
    memcpy(&creatorTokenID, dataAt, sizeof(creatorTokenID));
    dataAt += sizeof(creatorTokenID);

    uint32_t particleID;
    memcpy(&particleID, dataAt, sizeof(particleID));
    dataAt += sizeof(particleID);

    //qDebug() << "handleAddParticleResponse()... particleID=" << particleID << " creatorTokenID=" << creatorTokenID;

    // add our token to id mapping
    _tokenIDsToIDs[creatorTokenID] = particleID;
}

Particle::Particle() {
    rgbColor noColor = { 0, 0, 0 };
    init(glm::vec3(0,0,0), 0, noColor, glm::vec3(0,0,0),
            DEFAULT_GRAVITY, DEFAULT_DAMPING, DEFAULT_LIFETIME, NOT_IN_HAND, DEFAULT_SCRIPT, NEW_PARTICLE);
}

Particle::Particle(const ParticleID& particleID, const ParticleProperties& properties) {
    _id = particleID.id;
    _creatorTokenID = particleID.creatorTokenID;

    // init values with defaults before calling setProperties
    uint64_t now = usecTimestampNow();
    _lastEdited = now;
    _lastUpdated = now;
    _created = now; // will get updated as appropriate in setAge()

    _position = glm::vec3(0,0,0);
    _radius = 0;
    _mass = 1.0f;
    rgbColor noColor = { 0, 0, 0 };
    memcpy(_color, noColor, sizeof(_color));
    _velocity = glm::vec3(0,0,0);
    _damping = DEFAULT_DAMPING;
    _lifetime = DEFAULT_LIFETIME;
    _gravity = DEFAULT_GRAVITY;
    _script = DEFAULT_SCRIPT;
    _inHand = NOT_IN_HAND;
    _shouldDie = false;
    
    setProperties(properties);
}


Particle::~Particle() {
}

void Particle::init(glm::vec3 position, float radius, rgbColor color, glm::vec3 velocity, glm::vec3 gravity,
                    float damping, float lifetime, bool inHand, QString updateScript, uint32_t id) {
    if (id == NEW_PARTICLE) {
        _id = _nextID;
        _nextID++;
    } else {
        _id = id;
    }
    quint64 now = usecTimestampNow();
    _lastEdited = now;
    _lastUpdated = now;
    _created = now; // will get updated as appropriate in setAge()

    _position = position;
    _radius = radius;
    _mass = 1.0f;
    memcpy(_color, color, sizeof(_color));
    _velocity = velocity;
    _damping = damping;
    _lifetime = lifetime;
    _gravity = gravity;
    _script = updateScript;
    _inHand = inHand;
    _shouldDie = false;
}

void Particle::setMass(float value) {
    if (value > 0.0f) {
        _mass = value;
    }
}

bool Particle::appendParticleData(OctreePacketData* packetData) const {

    bool success = packetData->appendValue(getID());

    //printf("Particle::appendParticleData()... getID()=%d\n", getID());

    if (success) {
        success = packetData->appendValue(getAge());
    }
    if (success) {
        success = packetData->appendValue(getLastUpdated());
    }
    if (success) {
        success = packetData->appendValue(getLastEdited());
    }
    if (success) {
        success = packetData->appendValue(getRadius());
    }
    if (success) {
        success = packetData->appendPosition(getPosition());
    }
    if (success) {
        success = packetData->appendColor(getColor());
    }
    if (success) {
        success = packetData->appendValue(getVelocity());
    }
    if (success) {
        success = packetData->appendValue(getGravity());
    }
    if (success) {
        success = packetData->appendValue(getDamping());
    }
    if (success) {
        success = packetData->appendValue(getLifetime());
    }
    if (success) {
        success = packetData->appendValue(getInHand());
    }
    if (success) {
        success = packetData->appendValue(getShouldDie());
    }
    if (success) {
        uint16_t scriptLength = _script.size() + 1; // include NULL
        success = packetData->appendValue(scriptLength);
        if (success) {
            success = packetData->appendRawData((const unsigned char*)qPrintable(_script), scriptLength);
        }
    }

    // modelURL
    if (success) {
        uint16_t modelURLLength = _modelURL.size() + 1; // include NULL
        success = packetData->appendValue(modelURLLength);
        if (success) {
            success = packetData->appendRawData((const unsigned char*)qPrintable(_modelURL), modelURLLength);
        }
    }
    // modelTranslation
    if (success) {
        success = packetData->appendValue(getModelTranslation());
    }
    // modelRotation
    if (success) {
        success = packetData->appendValue(getModelRotation());
    }
    // modelScale
    if (success) {
        success = packetData->appendValue(getModelScale());
    }

    return success;
}

int Particle::expectedBytes() {
    int expectedBytes = sizeof(uint32_t) // id
                + sizeof(float) // age
                + sizeof(quint64) // last updated
                + sizeof(quint64) // lasted edited
                + sizeof(float) // radius
                + sizeof(glm::vec3) // position
                + sizeof(rgbColor) // color
                + sizeof(glm::vec3) // velocity
                + sizeof(glm::vec3) // gravity
                + sizeof(float) // damping
                + sizeof(float) // lifetime
                + sizeof(bool); // inhand
                // potentially more...
    return expectedBytes;
}

int Particle::readParticleDataFromBuffer(const unsigned char* data, int bytesLeftToRead, ReadBitstreamToTreeParams& args) {
    int bytesRead = 0;
    if (bytesLeftToRead >= expectedBytes()) {
        int clockSkew = args.sourceNode ? args.sourceNode->getClockSkewUsec() : 0;

        const unsigned char* dataAt = data;

        // id
        memcpy(&_id, dataAt, sizeof(_id));
        dataAt += sizeof(_id);
        bytesRead += sizeof(_id);

        // age
        float age;
        memcpy(&age, dataAt, sizeof(age));
        dataAt += sizeof(age);
        bytesRead += sizeof(age);
        setAge(age);

        // _lastUpdated
        memcpy(&_lastUpdated, dataAt, sizeof(_lastUpdated));
        dataAt += sizeof(_lastUpdated);
        bytesRead += sizeof(_lastUpdated);
        _lastUpdated -= clockSkew;

        // _lastEdited
        memcpy(&_lastEdited, dataAt, sizeof(_lastEdited));
        dataAt += sizeof(_lastEdited);
        bytesRead += sizeof(_lastEdited);
        _lastEdited -= clockSkew;

        // radius
        memcpy(&_radius, dataAt, sizeof(_radius));
        dataAt += sizeof(_radius);
        bytesRead += sizeof(_radius);

        // position
        memcpy(&_position, dataAt, sizeof(_position));
        dataAt += sizeof(_position);
        bytesRead += sizeof(_position);

        // color
        memcpy(_color, dataAt, sizeof(_color));
        dataAt += sizeof(_color);
        bytesRead += sizeof(_color);

        // velocity
        memcpy(&_velocity, dataAt, sizeof(_velocity));
        dataAt += sizeof(_velocity);
        bytesRead += sizeof(_velocity);

        // gravity
        memcpy(&_gravity, dataAt, sizeof(_gravity));
        dataAt += sizeof(_gravity);
        bytesRead += sizeof(_gravity);

        // damping
        memcpy(&_damping, dataAt, sizeof(_damping));
        dataAt += sizeof(_damping);
        bytesRead += sizeof(_damping);

        // lifetime
        memcpy(&_lifetime, dataAt, sizeof(_lifetime));
        dataAt += sizeof(_lifetime);
        bytesRead += sizeof(_lifetime);

        // inHand
        memcpy(&_inHand, dataAt, sizeof(_inHand));
        dataAt += sizeof(_inHand);
        bytesRead += sizeof(_inHand);

        // shouldDie
        memcpy(&_shouldDie, dataAt, sizeof(_shouldDie));
        dataAt += sizeof(_shouldDie);
        bytesRead += sizeof(_shouldDie);

        // script
        uint16_t scriptLength;
        memcpy(&scriptLength, dataAt, sizeof(scriptLength));
        dataAt += sizeof(scriptLength);
        bytesRead += sizeof(scriptLength);
        QString tempString((const char*)dataAt);
        _script = tempString;
        dataAt += scriptLength;
        bytesRead += scriptLength;

        // modelURL
        uint16_t modelURLLength;
        memcpy(&modelURLLength, dataAt, sizeof(modelURLLength));
        dataAt += sizeof(modelURLLength);
        bytesRead += sizeof(modelURLLength);
        QString modelURLString((const char*)dataAt);
        _modelURL = modelURLString;
        dataAt += modelURLLength;
        bytesRead += modelURLLength;

        // modelTranslation
        memcpy(&_modelTranslation, dataAt, sizeof(_modelTranslation));
        dataAt += sizeof(_modelTranslation);
        bytesRead += sizeof(_modelTranslation);

        // modelRotation
        int bytes = unpackOrientationQuatFromBytes(dataAt, _modelRotation);
        dataAt += bytes;
        bytesRead += bytes;

        // modelScale
        memcpy(&_modelScale, dataAt, sizeof(_modelScale));
        dataAt += sizeof(_modelScale);
        bytesRead += sizeof(_modelScale);

        //printf("Particle::readParticleDataFromBuffer()... "); debugDump();
    }
    return bytesRead;
}

Particle Particle::fromEditPacket(const unsigned char* data, int length, int& processedBytes, ParticleTree* tree, bool& valid) {

    Particle newParticle; // id and _lastUpdated will get set here...
    const unsigned char* dataAt = data;
    processedBytes = 0;

    // the first part of the data is our octcode...
    int octets = numberOfThreeBitSectionsInCode(data);
    int lengthOfOctcode = bytesRequiredForCodeLength(octets);

    // we don't actually do anything with this octcode...
    dataAt += lengthOfOctcode;
    processedBytes += lengthOfOctcode;

    // id
    uint32_t editID;
    memcpy(&editID, dataAt, sizeof(editID));
    dataAt += sizeof(editID);
    processedBytes += sizeof(editID);

    //qDebug() << "editID:" << editID;

    bool isNewParticle = (editID == NEW_PARTICLE);

    // special case for handling "new" particles
    if (isNewParticle) {
        // If this is a NEW_PARTICLE, then we assume that there's an additional uint32_t creatorToken, that
        // we want to send back to the creator as an map to the actual id
        uint32_t creatorTokenID;
        memcpy(&creatorTokenID, dataAt, sizeof(creatorTokenID));
        dataAt += sizeof(creatorTokenID);
        processedBytes += sizeof(creatorTokenID);

        newParticle.setCreatorTokenID(creatorTokenID);
        newParticle._newlyCreated = true;
        newParticle.setAge(0); // this guy is new!

    } else {
        // look up the existing particle
        const Particle* existingParticle = tree->findParticleByID(editID, true);

        // copy existing properties before over-writing with new properties
        if (existingParticle) {
            newParticle = *existingParticle;
        } else {
            // the user attempted to edit a particle that doesn't exist
            qDebug() << "user attempted to edit a particle that doesn't exist...";
            valid = false;
            return newParticle;
        }
        newParticle._id = editID;
        newParticle._newlyCreated = false;
    }
    
    // if we got this far, then our result will be valid
    valid = true;
    

    // lastEdited
    memcpy(&newParticle._lastEdited, dataAt, sizeof(newParticle._lastEdited));
    dataAt += sizeof(newParticle._lastEdited);
    processedBytes += sizeof(newParticle._lastEdited);

    // All of the remaining items are optional, and may or may not be included based on their included values in the
    // properties included bits
    uint16_t packetContainsBits = 0;
    if (!isNewParticle) {
        memcpy(&packetContainsBits, dataAt, sizeof(packetContainsBits));
        dataAt += sizeof(packetContainsBits);
        processedBytes += sizeof(packetContainsBits);
        //qDebug() << "packetContainsBits:" << packetContainsBits;
    }


    // radius
    if (isNewParticle || ((packetContainsBits & CONTAINS_RADIUS) == CONTAINS_RADIUS)) {
        memcpy(&newParticle._radius, dataAt, sizeof(newParticle._radius));
        dataAt += sizeof(newParticle._radius);
        processedBytes += sizeof(newParticle._radius);
    }

    // position
    if (isNewParticle || ((packetContainsBits & CONTAINS_POSITION) == CONTAINS_POSITION)) {
        memcpy(&newParticle._position, dataAt, sizeof(newParticle._position));
        dataAt += sizeof(newParticle._position);
        processedBytes += sizeof(newParticle._position);
    }

    // color
    if (isNewParticle || ((packetContainsBits & CONTAINS_COLOR) == CONTAINS_COLOR)) {
        memcpy(newParticle._color, dataAt, sizeof(newParticle._color));
        dataAt += sizeof(newParticle._color);
        processedBytes += sizeof(newParticle._color);
    }

    // velocity
    if (isNewParticle || ((packetContainsBits & CONTAINS_VELOCITY) == CONTAINS_VELOCITY)) {
        memcpy(&newParticle._velocity, dataAt, sizeof(newParticle._velocity));
        dataAt += sizeof(newParticle._velocity);
        processedBytes += sizeof(newParticle._velocity);
    }

    // gravity
    if (isNewParticle || ((packetContainsBits & CONTAINS_GRAVITY) == CONTAINS_GRAVITY)) {
        memcpy(&newParticle._gravity, dataAt, sizeof(newParticle._gravity));
        dataAt += sizeof(newParticle._gravity);
        processedBytes += sizeof(newParticle._gravity);
    }

    // damping
    if (isNewParticle || ((packetContainsBits & CONTAINS_DAMPING) == CONTAINS_DAMPING)) {
        memcpy(&newParticle._damping, dataAt, sizeof(newParticle._damping));
        dataAt += sizeof(newParticle._damping);
        processedBytes += sizeof(newParticle._damping);
    }

    // lifetime
    if (isNewParticle || ((packetContainsBits & CONTAINS_LIFETIME) == CONTAINS_LIFETIME)) {
        memcpy(&newParticle._lifetime, dataAt, sizeof(newParticle._lifetime));
        dataAt += sizeof(newParticle._lifetime);
        processedBytes += sizeof(newParticle._lifetime);
    }

    // TODO: make inHand and shouldDie into single bits
    // inHand
    if (isNewParticle || ((packetContainsBits & CONTAINS_INHAND) == CONTAINS_INHAND)) {
        memcpy(&newParticle._inHand, dataAt, sizeof(newParticle._inHand));
        dataAt += sizeof(newParticle._inHand);
        processedBytes += sizeof(newParticle._inHand);
    }

    // shouldDie
    if (isNewParticle || ((packetContainsBits & CONTAINS_SHOULDDIE) == CONTAINS_SHOULDDIE)) {
        memcpy(&newParticle._shouldDie, dataAt, sizeof(newParticle._shouldDie));
        dataAt += sizeof(newParticle._shouldDie);
        processedBytes += sizeof(newParticle._shouldDie);
    }

    // script
    if (isNewParticle || ((packetContainsBits & CONTAINS_SCRIPT) == CONTAINS_SCRIPT)) {
        uint16_t scriptLength;
        memcpy(&scriptLength, dataAt, sizeof(scriptLength));
        dataAt += sizeof(scriptLength);
        processedBytes += sizeof(scriptLength);
        QString tempString((const char*)dataAt);
        newParticle._script = tempString;
        dataAt += scriptLength;
        processedBytes += scriptLength;
    }

    // modelURL
    if (isNewParticle || ((packetContainsBits & CONTAINS_MODEL_URL) == CONTAINS_MODEL_URL)) {
        uint16_t modelURLLength;
        memcpy(&modelURLLength, dataAt, sizeof(modelURLLength));
        dataAt += sizeof(modelURLLength);
        processedBytes += sizeof(modelURLLength);
        QString tempString((const char*)dataAt);
        newParticle._modelURL = tempString;
        dataAt += modelURLLength;
        processedBytes += modelURLLength;
    }

    // modelTranslation
    if (isNewParticle || ((packetContainsBits & CONTAINS_MODEL_TRANSLATION) == CONTAINS_MODEL_TRANSLATION)) {
        memcpy(&newParticle._modelTranslation, dataAt, sizeof(newParticle._modelTranslation));
        dataAt += sizeof(newParticle._modelTranslation);
        processedBytes += sizeof(newParticle._modelTranslation);
    }

    // modelRotation
    if (isNewParticle || ((packetContainsBits & CONTAINS_MODEL_ROTATION) == CONTAINS_MODEL_ROTATION)) {
        int bytes = unpackOrientationQuatFromBytes(dataAt, newParticle._modelRotation);
        dataAt += bytes;
        processedBytes += bytes;
    }

    // modelScale
    if (isNewParticle || ((packetContainsBits & CONTAINS_MODEL_SCALE) == CONTAINS_MODEL_SCALE)) {
        memcpy(&newParticle._modelScale, dataAt, sizeof(newParticle._modelScale));
        dataAt += sizeof(newParticle._modelScale);
        processedBytes += sizeof(newParticle._modelScale);
    }

    const bool wantDebugging = false;
    if (wantDebugging) {
        qDebug("Particle::fromEditPacket()...");
        qDebug() << "   Particle id in packet:" << editID;
        //qDebug() << "    position: " << newParticle._position;
        newParticle.debugDump();
    }

    return newParticle;
}

void Particle::debugDump() const {
    printf("Particle id  :%u\n", _id);
    printf(" age:%f\n", getAge());
    printf(" edited ago:%f\n", getEditedAgo());
    printf(" should die:%s\n", debug::valueOf(getShouldDie()));
    printf(" position:%f,%f,%f\n", _position.x, _position.y, _position.z);
    printf(" radius:%f\n", getRadius());
    printf(" velocity:%f,%f,%f\n", _velocity.x, _velocity.y, _velocity.z);
    printf(" gravity:%f,%f,%f\n", _gravity.x, _gravity.y, _gravity.z);
    printf(" color:%d,%d,%d\n", _color[0], _color[1], _color[2]);
}

bool Particle::encodeParticleEditMessageDetails(PacketType command, ParticleID id, const ParticleProperties& properties,
        unsigned char* bufferOut, int sizeIn, int& sizeOut) {

    bool success = true; // assume the best
    unsigned char* copyAt = bufferOut;
    sizeOut = 0;

    // get the octal code for the particle

    // this could be a problem if the caller doesn't include position....
    glm::vec3 rootPosition(0);
    float rootScale = 0.5f;
    unsigned char* octcode = pointToOctalCode(rootPosition.x, rootPosition.y, rootPosition.z, rootScale);

    // TODO: Consider this old code... including the correct octree for where the particle will go matters for 
    // particle servers with different jurisdictions, but for now, we'll send everything to the root, since the 
    // tree does the right thing...
    //
    //unsigned char* octcode = pointToOctalCode(details[i].position.x, details[i].position.y,
    //                                          details[i].position.z, details[i].radius);

    int octets = numberOfThreeBitSectionsInCode(octcode);
    int lengthOfOctcode = bytesRequiredForCodeLength(octets);

    // add it to our message
    memcpy(copyAt, octcode, lengthOfOctcode);
    copyAt += lengthOfOctcode;
    sizeOut += lengthOfOctcode;

    // Now add our edit content details...
    bool isNewParticle = (id.id == NEW_PARTICLE);

    // id
    memcpy(copyAt, &id.id, sizeof(id.id));
    copyAt += sizeof(id.id);
    sizeOut += sizeof(id.id);

    // special case for handling "new" particles
    if (isNewParticle) {
        // If this is a NEW_PARTICLE, then we assume that there's an additional uint32_t creatorToken, that
        // we want to send back to the creator as an map to the actual id
        memcpy(copyAt, &id.creatorTokenID, sizeof(id.creatorTokenID));
        copyAt += sizeof(id.creatorTokenID);
        sizeOut += sizeof(id.creatorTokenID);
    }
    
    // lastEdited
    quint64 lastEdited = properties.getLastEdited();
    memcpy(copyAt, &lastEdited, sizeof(lastEdited));
    copyAt += sizeof(lastEdited);
    sizeOut += sizeof(lastEdited);
    
    // For new particles, all remaining items are mandatory, for an edited particle, All of the remaining items are
    // optional, and may or may not be included based on their included values in the properties included bits
    uint16_t packetContainsBits = properties.getChangedBits();
    if (!isNewParticle) {
        memcpy(copyAt, &packetContainsBits, sizeof(packetContainsBits));
        copyAt += sizeof(packetContainsBits);
        sizeOut += sizeof(packetContainsBits);
    }

    // radius
    if (isNewParticle || ((packetContainsBits & CONTAINS_RADIUS) == CONTAINS_RADIUS)) {
        float radius = properties.getRadius() / (float) TREE_SCALE;
        memcpy(copyAt, &radius, sizeof(radius));
        copyAt += sizeof(radius);
        sizeOut += sizeof(radius);
    }

    // position
    if (isNewParticle || ((packetContainsBits & CONTAINS_POSITION) == CONTAINS_POSITION)) {
        glm::vec3 position = properties.getPosition() / (float)TREE_SCALE;
        memcpy(copyAt, &position, sizeof(position));
        copyAt += sizeof(position);
        sizeOut += sizeof(position);
    }

    // color
    if (isNewParticle || ((packetContainsBits & CONTAINS_COLOR) == CONTAINS_COLOR)) {
        rgbColor color = { properties.getColor().red, properties.getColor().green, properties.getColor().blue };
        memcpy(copyAt, color, sizeof(color));
        copyAt += sizeof(color);
        sizeOut += sizeof(color);
    }

    // velocity
    if (isNewParticle || ((packetContainsBits & CONTAINS_VELOCITY) == CONTAINS_VELOCITY)) {
        glm::vec3 velocity = properties.getVelocity() / (float)TREE_SCALE;
        memcpy(copyAt, &velocity, sizeof(velocity));
        copyAt += sizeof(velocity);
        sizeOut += sizeof(velocity);
    }

    // gravity
    if (isNewParticle || ((packetContainsBits & CONTAINS_GRAVITY) == CONTAINS_GRAVITY)) {
        glm::vec3 gravity = properties.getGravity() / (float)TREE_SCALE;
        memcpy(copyAt, &gravity, sizeof(gravity));
        copyAt += sizeof(gravity);
        sizeOut += sizeof(gravity);
    }

    // damping
    if (isNewParticle || ((packetContainsBits & CONTAINS_DAMPING) == CONTAINS_DAMPING)) {
        float damping = properties.getDamping();
        memcpy(copyAt, &damping, sizeof(damping));
        copyAt += sizeof(damping);
        sizeOut += sizeof(damping);
    }

    // lifetime
    if (isNewParticle || ((packetContainsBits & CONTAINS_LIFETIME) == CONTAINS_LIFETIME)) {
        float lifetime = properties.getLifetime();
        memcpy(copyAt, &lifetime, sizeof(lifetime));
        copyAt += sizeof(lifetime);
        sizeOut += sizeof(lifetime);
    }

    // inHand
    if (isNewParticle || ((packetContainsBits & CONTAINS_INHAND) == CONTAINS_INHAND)) {
        bool inHand = properties.getInHand();
        memcpy(copyAt, &inHand, sizeof(inHand));
        copyAt += sizeof(inHand);
        sizeOut += sizeof(inHand);
    }

    // shoulDie
    if (isNewParticle || ((packetContainsBits & CONTAINS_SHOULDDIE) == CONTAINS_SHOULDDIE)) {
        bool shouldDie = properties.getShouldDie();
        memcpy(copyAt, &shouldDie, sizeof(shouldDie));
        copyAt += sizeof(shouldDie);
        sizeOut += sizeof(shouldDie);
    }

    // script
    if (isNewParticle || ((packetContainsBits & CONTAINS_SCRIPT) == CONTAINS_SCRIPT)) {
        uint16_t scriptLength = properties.getScript().size() + 1;
        memcpy(copyAt, &scriptLength, sizeof(scriptLength));
        copyAt += sizeof(scriptLength);
        sizeOut += sizeof(scriptLength);
        memcpy(copyAt, qPrintable(properties.getScript()), scriptLength);
        copyAt += scriptLength;
        sizeOut += scriptLength;
    }

    // modelURL
    if (isNewParticle || ((packetContainsBits & CONTAINS_MODEL_URL) == CONTAINS_MODEL_URL)) {
        uint16_t urlLength = properties.getModelURL().size() + 1;
        memcpy(copyAt, &urlLength, sizeof(urlLength));
        copyAt += sizeof(urlLength);
        sizeOut += sizeof(urlLength);
        memcpy(copyAt, qPrintable(properties.getModelURL()), urlLength);
        copyAt += urlLength;
        sizeOut += urlLength;
    }

    // modelTranslation
    if (isNewParticle || ((packetContainsBits & CONTAINS_MODEL_TRANSLATION) == CONTAINS_MODEL_TRANSLATION)) {
        glm::vec3 modelTranslation = properties.getModelTranslation(); // should this be relative to TREE_SCALE??
        memcpy(copyAt, &modelTranslation, sizeof(modelTranslation));
        copyAt += sizeof(modelTranslation);
        sizeOut += sizeof(modelTranslation);
    }

    // modelRotation
    if (isNewParticle || ((packetContainsBits & CONTAINS_MODEL_ROTATION) == CONTAINS_MODEL_ROTATION)) {
        int bytes = packOrientationQuatToBytes(copyAt, properties.getModelRotation());
        copyAt += bytes;
        sizeOut += bytes;
    }

    // modelScale
    if (isNewParticle || ((packetContainsBits & CONTAINS_MODEL_SCALE) == CONTAINS_MODEL_SCALE)) {
        float modelScale = properties.getModelScale();
        memcpy(copyAt, &modelScale, sizeof(modelScale));
        copyAt += sizeof(modelScale);
        sizeOut += sizeof(modelScale);
    }

    bool wantDebugging = false;
    if (wantDebugging) {
        printf("encodeParticleEditMessageDetails()....\n");
        printf("Particle id  :%u\n", id.id);
        printf(" nextID:%u\n", _nextID);
    }

    // cleanup
    delete[] octcode;
    
    //qDebug() << "encoding... sizeOut:" << sizeOut;

    return success;
}

// adjust any internal timestamps to fix clock skew for this server
void Particle::adjustEditPacketForClockSkew(unsigned char* codeColorBuffer, ssize_t length, int clockSkew) {
    unsigned char* dataAt = codeColorBuffer;
    int octets = numberOfThreeBitSectionsInCode(dataAt);
    int lengthOfOctcode = bytesRequiredForCodeLength(octets);
    dataAt += lengthOfOctcode;

    // id
    uint32_t id;
    memcpy(&id, dataAt, sizeof(id));
    dataAt += sizeof(id);
    // special case for handling "new" particles
    if (id == NEW_PARTICLE) {
        // If this is a NEW_PARTICLE, then we assume that there's an additional uint32_t creatorToken, that
        // we want to send back to the creator as an map to the actual id
        dataAt += sizeof(uint32_t);
    }

    // lastEdited
    quint64 lastEditedInLocalTime;
    memcpy(&lastEditedInLocalTime, dataAt, sizeof(lastEditedInLocalTime));
    quint64 lastEditedInServerTime = lastEditedInLocalTime + clockSkew;
    memcpy(dataAt, &lastEditedInServerTime, sizeof(lastEditedInServerTime));
    const bool wantDebug = false;
    if (wantDebug) {
        qDebug("Particle::adjustEditPacketForClockSkew()...");
        qDebug() << "     lastEditedInLocalTime: " << lastEditedInLocalTime;
        qDebug() << "                 clockSkew: " << clockSkew;
        qDebug() << "    lastEditedInServerTime: " << lastEditedInServerTime;
    }
}

// HALTING_* params are determined using expected acceleration of gravity over some timescale.  
// This is a HACK for particles that bounce in a 1.0 gravitational field and should eventually be made more universal.
const float HALTING_PARTICLE_PERIOD = 0.0167f;  // ~1/60th of a second
const float HALTING_PARTICLE_SPEED = 9.8 * HALTING_PARTICLE_PERIOD / (float)(TREE_SCALE);

void Particle::applyHardCollision(const CollisionInfo& collisionInfo) {
    //
    //  Update the particle in response to a hard collision.  Position will be reset exactly
    //  to outside the colliding surface.  Velocity will be modified according to elasticity.
    //
    //  if elasticity = 0.0, collision is inelastic (vel normal to collision is lost)
    //  if elasticity = 1.0, collision is 100% elastic.
    //
    glm::vec3 position = getPosition();
    glm::vec3 velocity = getVelocity();

    const float EPSILON = 0.0f;
    glm::vec3 relativeVelocity = collisionInfo._addedVelocity - velocity;
    float velocityDotPenetration = glm::dot(relativeVelocity, collisionInfo._penetration);
    if (velocityDotPenetration < EPSILON) {
        // particle is moving into collision surface
        //
        // TODO: do something smarter here by comparing the mass of the particle vs that of the other thing 
        // (other's mass could be stored in the Collision Info).  The smaller mass should surrender more 
        // position offset and should slave more to the other's velocity in the static-friction case.
        position -= collisionInfo._penetration;

        if (glm::length(relativeVelocity) < HALTING_PARTICLE_SPEED) {
            // static friction kicks in and particle moves with colliding object
            velocity = collisionInfo._addedVelocity;
        } else {
            glm::vec3 direction = glm::normalize(collisionInfo._penetration);
            velocity += glm::dot(relativeVelocity, direction) * (1.0f + collisionInfo._elasticity) * direction;    // dynamic reflection
            velocity += glm::clamp(collisionInfo._damping, 0.0f, 1.0f) * (relativeVelocity - glm::dot(relativeVelocity, direction) * direction);   // dynamic friction
        }
    }

    // change the local particle too...
    setPosition(position);
    setVelocity(velocity);
}

// MIN_VALID_SPEED is obtained by computing speed gained at one gravity during the shortest expected frame period
// This is a HACK for particles that bounce in a 1.0 gravitational field and should eventually be made more universal.
const float MIN_EXPECTED_FRAME_PERIOD = 0.005f;  // 1/200th of a second
const float MIN_VALID_SPEED = 9.8 * MIN_EXPECTED_FRAME_PERIOD / (float)(TREE_SCALE);

void Particle::update(const quint64& now) {
    float timeElapsed = (float)(now - _lastUpdated) / (float)(USECS_PER_SECOND);
    _lastUpdated = now;

    // calculate our default shouldDie state... then allow script to change it if it wants...
    float speed = glm::length(_velocity);
    bool isStopped = (speed < MIN_VALID_SPEED);
    const quint64 REALLY_OLD = 30 * USECS_PER_SECOND; // 30 seconds
    bool isReallyOld = ((now - _created) > REALLY_OLD);
    bool isInHand = getInHand();
    bool shouldDie = (getAge() > getLifetime()) || getShouldDie() || (!isInHand && isStopped && isReallyOld);
    setShouldDie(shouldDie);

    runUpdateScript(); // allow the javascript to alter our state

    // If the ball is in hand, it doesn't move or have gravity effect it
    if (!isInHand) {
        _position += _velocity * timeElapsed;

        // handle bounces off the ground...
        if (_position.y <= 0) {
            _velocity = _velocity * glm::vec3(1,-1,1);
            _position.y = 0;
        }

        // handle gravity....
        _velocity += _gravity * timeElapsed;

        // handle damping
        glm::vec3 dampingResistance = _velocity * _damping;
        _velocity -= dampingResistance * timeElapsed;
        //printf("applying damping to Particle timeElapsed=%f\n",timeElapsed);
    }
}

void Particle::runUpdateScript() {
    if (!_script.isEmpty()) {
        ScriptEngine engine(_script); // no menu or controller interface...

        if (_voxelEditSender) {
            engine.getVoxelsScriptingInterface()->setPacketSender(_voxelEditSender);
        }
        if (_particleEditSender) {
            engine.getParticlesScriptingInterface()->setPacketSender(_particleEditSender);
        }

        // Add the Particle object
        ParticleScriptObject particleScriptable(this);
        engine.registerGlobalObject("Particle", &particleScriptable);

        // init and evaluate the script, but return so we can emit the collision
        engine.evaluate();

        particleScriptable.emitUpdate();

        // it seems like we may need to send out particle edits if the state of our particle was changed.

        if (_voxelEditSender) {
            _voxelEditSender->releaseQueuedMessages();
        }
        if (_particleEditSender) {
            _particleEditSender->releaseQueuedMessages();
        }
    }
}

void Particle::collisionWithParticle(Particle* other) {
    if (!_script.isEmpty()) {
        ScriptEngine engine(_script); // no menu or controller interface...

        if (_voxelEditSender) {
            engine.getVoxelsScriptingInterface()->setPacketSender(_voxelEditSender);
        }
        if (_particleEditSender) {
            engine.getParticlesScriptingInterface()->setPacketSender(_particleEditSender);
        }

        // Add the Particle object
        ParticleScriptObject particleScriptable(this);
        engine.registerGlobalObject("Particle", &particleScriptable);

        // init and evaluate the script, but return so we can emit the collision
        engine.evaluate();

        ParticleScriptObject otherParticleScriptable(other);
        particleScriptable.emitCollisionWithParticle(&otherParticleScriptable);

        // it seems like we may need to send out particle edits if the state of our particle was changed.

        if (_voxelEditSender) {
            _voxelEditSender->releaseQueuedMessages();
        }
        if (_particleEditSender) {
            _particleEditSender->releaseQueuedMessages();
        }
    }
}

void Particle::collisionWithVoxel(VoxelDetail* voxelDetails) {
    if (!_script.isEmpty()) {

        ScriptEngine engine(_script); // no menu or controller interface...

        // setup the packet senders and jurisdiction listeners of the script engine's scripting interfaces so
        // we can use the same ones as our context.
        if (_voxelEditSender) {
            engine.getVoxelsScriptingInterface()->setPacketSender(_voxelEditSender);
        }
        if (_particleEditSender) {
            engine.getParticlesScriptingInterface()->setPacketSender(_particleEditSender);
        }

        // Add the Particle object
        ParticleScriptObject particleScriptable(this);
        engine.registerGlobalObject("Particle", &particleScriptable);

        // init and evaluate the script, but return so we can emit the collision
        engine.evaluate();

        VoxelDetailScriptObject voxelDetailsScriptable(voxelDetails);
        particleScriptable.emitCollisionWithVoxel(&voxelDetailsScriptable);

        // it seems like we may need to send out particle edits if the state of our particle was changed.

        if (_voxelEditSender) {
            _voxelEditSender->releaseQueuedMessages();
        }
        if (_particleEditSender) {
            _particleEditSender->releaseQueuedMessages();
        }
    }
}



void Particle::setAge(float age) {
    quint64 ageInUsecs = age * USECS_PER_SECOND;
    _created = usecTimestampNow() - ageInUsecs;
}

void Particle::copyChangedProperties(const Particle& other) {
    float age = getAge();
    *this = other;
    setAge(age);
}

ParticleProperties Particle::getProperties() const {
    ParticleProperties properties;
    properties.copyFromParticle(*this);
    return properties;
}

void Particle::setProperties(const ParticleProperties& properties) {
    properties.copyToParticle(*this);
}

ParticleProperties::ParticleProperties() :
    _position(0),
    _color(),
    _radius(DEFAULT_RADIUS),
    _velocity(0),
    _gravity(DEFAULT_GRAVITY),
    _damping(DEFAULT_DAMPING),
    _lifetime(DEFAULT_LIFETIME),
    _script(""),
    _inHand(false),
    _shouldDie(false),
    _modelURL(""),
    _modelTranslation(DEFAULT_MODEL_TRANSLATION),
    _modelRotation(DEFAULT_MODEL_ROTATION),
    _modelScale(DEFAULT_MODEL_SCALE),

    _id(UNKNOWN_PARTICLE_ID),
    _idSet(false),
    _lastEdited(usecTimestampNow()),

    _positionChanged(false),
    _colorChanged(false),
    _radiusChanged(false),
    _velocityChanged(false),
    _gravityChanged(false),
    _dampingChanged(false),
    _lifetimeChanged(false),
    _scriptChanged(false),
    _inHandChanged(false),
    _shouldDieChanged(false),
    _modelURLChanged(false),
    _modelTranslationChanged(false),
    _modelRotationChanged(false),
    _modelScaleChanged(false),
    _defaultSettings(true)
{
}


uint16_t ParticleProperties::getChangedBits() const {
    uint16_t changedBits = 0;
    if (_radiusChanged) {
        changedBits += CONTAINS_RADIUS;
    }

    if (_positionChanged) {
        changedBits += CONTAINS_POSITION;
    }

    if (_colorChanged) {
        changedBits += CONTAINS_COLOR;
    }

    if (_velocityChanged) {
        changedBits += CONTAINS_VELOCITY;
    }

    if (_gravityChanged) {
        changedBits += CONTAINS_GRAVITY;
    }

    if (_dampingChanged) {
        changedBits += CONTAINS_DAMPING;
    }

    if (_lifetimeChanged) {
        changedBits += CONTAINS_LIFETIME;
    }

    if (_inHandChanged) {
        changedBits += CONTAINS_INHAND;
    }

    if (_scriptChanged) {
        changedBits += CONTAINS_SCRIPT;
    }

    if (_shouldDieChanged) {
        changedBits += CONTAINS_SHOULDDIE;
    }

    if (_modelURLChanged) {
        changedBits += CONTAINS_MODEL_URL;
    }

    if (_modelTranslationChanged) {
        changedBits += CONTAINS_MODEL_TRANSLATION;
    }

    if (_modelRotationChanged) {
        changedBits += CONTAINS_MODEL_ROTATION;
    }

    if (_modelScaleChanged) {
        changedBits += CONTAINS_MODEL_SCALE;
    }

    return changedBits;
}


QScriptValue ParticleProperties::copyToScriptValue(QScriptEngine* engine) const {
    QScriptValue properties = engine->newObject();

    QScriptValue position = vec3toScriptValue(engine, _position);
    properties.setProperty("position", position);

    QScriptValue color = xColorToScriptValue(engine, _color);
    properties.setProperty("color", color);

    properties.setProperty("radius", _radius);

    QScriptValue velocity = vec3toScriptValue(engine, _velocity);
    properties.setProperty("velocity", velocity);

    QScriptValue gravity = vec3toScriptValue(engine, _gravity);
    properties.setProperty("gravity", gravity);

    properties.setProperty("damping", _damping);
    properties.setProperty("lifetime", _lifetime);
    properties.setProperty("script", _script);
    properties.setProperty("inHand", _inHand);
    properties.setProperty("shouldDie", _shouldDie);

    properties.setProperty("modelURL", _modelURL);

    QScriptValue modelTranslation = vec3toScriptValue(engine, _modelTranslation);
    properties.setProperty("modelTranslation", modelTranslation);

    QScriptValue modelRotation = quatToScriptValue(engine, _modelRotation);
    properties.setProperty("modelRotation", modelRotation);

    properties.setProperty("modelScale", _modelScale);


    if (_idSet) {
        properties.setProperty("id", _id);
        properties.setProperty("isKnownID", (_id == UNKNOWN_PARTICLE_ID));
    }

    return properties;
}

void ParticleProperties::copyFromScriptValue(const QScriptValue &object) {

    QScriptValue position = object.property("position");
    if (position.isValid()) {
        QScriptValue x = position.property("x");
        QScriptValue y = position.property("y");
        QScriptValue z = position.property("z");
        if (x.isValid() && y.isValid() && z.isValid()) {
            glm::vec3 newPosition;
            newPosition.x = x.toVariant().toFloat();
            newPosition.y = y.toVariant().toFloat();
            newPosition.z = z.toVariant().toFloat();
            if (_defaultSettings || newPosition != _position) {
                _position = newPosition;
                _positionChanged = true;
            }
        }
    }

    QScriptValue color = object.property("color");
    if (color.isValid()) {
        QScriptValue red = color.property("red");
        QScriptValue green = color.property("green");
        QScriptValue blue = color.property("blue");
        if (red.isValid() && green.isValid() && blue.isValid()) {
            xColor newColor;
            newColor.red = red.toVariant().toInt();
            newColor.green = green.toVariant().toInt();
            newColor.blue = blue.toVariant().toInt();
            if (_defaultSettings || (newColor.red != _color.red ||
                newColor.green != _color.green ||
                newColor.blue != _color.blue)) {
                _color = newColor;
                _colorChanged = true;
            }
        }
    }

    QScriptValue radius = object.property("radius");
    if (radius.isValid()) {
        float newRadius;
        newRadius = radius.toVariant().toFloat();
        if (_defaultSettings || newRadius != _radius) {
            _radius = newRadius;
            _radiusChanged = true;
        }
    }

    QScriptValue velocity = object.property("velocity");
    if (velocity.isValid()) {
        QScriptValue x = velocity.property("x");
        QScriptValue y = velocity.property("y");
        QScriptValue z = velocity.property("z");
        if (x.isValid() && y.isValid() && z.isValid()) {
            glm::vec3 newVelocity;
            newVelocity.x = x.toVariant().toFloat();
            newVelocity.y = y.toVariant().toFloat();
            newVelocity.z = z.toVariant().toFloat();
            if (_defaultSettings || newVelocity != _velocity) {
                _velocity = newVelocity;
                _velocityChanged = true;
            }
        }
    }

    QScriptValue gravity = object.property("gravity");
    if (gravity.isValid()) {
        QScriptValue x = gravity.property("x");
        QScriptValue y = gravity.property("y");
        QScriptValue z = gravity.property("z");
        if (x.isValid() && y.isValid() && z.isValid()) {
            glm::vec3 newGravity;
            newGravity.x = x.toVariant().toFloat();
            newGravity.y = y.toVariant().toFloat();
            newGravity.z = z.toVariant().toFloat();
            if (_defaultSettings || newGravity != _gravity) {
                _gravity = newGravity;
                _gravityChanged = true;
            }
        }
    }

    QScriptValue damping = object.property("damping");
    if (damping.isValid()) {
        float newDamping;
        newDamping = damping.toVariant().toFloat();
        if (_defaultSettings || newDamping != _damping) {
            _damping = newDamping;
            _dampingChanged = true;
        }
    }

    QScriptValue lifetime = object.property("lifetime");
    if (lifetime.isValid()) {
        float newLifetime;
        newLifetime = lifetime.toVariant().toFloat();
        if (_defaultSettings || newLifetime != _lifetime) {
            _lifetime = newLifetime;
            _lifetimeChanged = true;
        }
    }

    QScriptValue script = object.property("script");
    if (script.isValid()) {
        QString newScript;
        newScript = script.toVariant().toString();
        if (_defaultSettings || newScript != _script) {
            _script = newScript;
            _scriptChanged = true;
        }
    }

    QScriptValue inHand = object.property("inHand");
    if (inHand.isValid()) {
        bool newInHand;
        newInHand = inHand.toVariant().toBool();
        if (_defaultSettings || newInHand != _inHand) {
            _inHand = newInHand;
            _inHandChanged = true;
        }
    }

    QScriptValue shouldDie = object.property("shouldDie");
    if (shouldDie.isValid()) {
        bool newShouldDie;
        newShouldDie = shouldDie.toVariant().toBool();
        if (_defaultSettings || newShouldDie != _shouldDie) {
            _shouldDie = newShouldDie;
            _shouldDieChanged = true;
        }
    }

    QScriptValue modelURL = object.property("modelURL");
    if (modelURL.isValid()) {
        QString newModelURL;
        newModelURL = modelURL.toVariant().toString();
        if (_defaultSettings || newModelURL != _modelURL) {
            _modelURL = newModelURL;
            _modelURLChanged = true;
        }
    }
    
    QScriptValue modelTranslation = object.property("modelTranslation");
    if (modelTranslation.isValid()) {
        QScriptValue x = modelTranslation.property("x");
        QScriptValue y = modelTranslation.property("y");
        QScriptValue z = modelTranslation.property("z");
        if (x.isValid() && y.isValid() && z.isValid()) {
            glm::vec3 newModelTranslation;
            newModelTranslation.x = x.toVariant().toFloat();
            newModelTranslation.y = y.toVariant().toFloat();
            newModelTranslation.z = z.toVariant().toFloat();
            if (_defaultSettings || newModelTranslation != _modelTranslation) {
                _modelTranslation = newModelTranslation;
                _modelTranslationChanged = true;
            }
        }
    }

    
    QScriptValue modelRotation = object.property("modelRotation");
    if (modelRotation.isValid()) {
        QScriptValue x = modelRotation.property("x");
        QScriptValue y = modelRotation.property("y");
        QScriptValue z = modelRotation.property("z");
        QScriptValue w = modelRotation.property("w");
        if (x.isValid() && y.isValid() && z.isValid() && w.isValid()) {
            glm::quat newModelRotation;
            newModelRotation.x = x.toVariant().toFloat();
            newModelRotation.y = y.toVariant().toFloat();
            newModelRotation.z = z.toVariant().toFloat();
            newModelRotation.w = w.toVariant().toFloat();
            if (_defaultSettings || newModelRotation != _modelRotation) {
                _modelRotation = newModelRotation;
                _modelRotationChanged = true;
            }
        }
    }

    QScriptValue modelScale = object.property("modelScale");
    if (modelScale.isValid()) {
        float newModelScale;
        newModelScale = modelScale.toVariant().toFloat();
        if (_defaultSettings || newModelScale != _modelScale) {
            _modelScale = newModelScale;
            _modelScaleChanged = true;
        }
    }

    _lastEdited = usecTimestampNow();
}

void ParticleProperties::copyToParticle(Particle& particle) const {
    bool somethingChanged = false;
    if (_positionChanged) {
        particle.setPosition(_position / (float) TREE_SCALE);
        somethingChanged = true;
    }

    if (_colorChanged) {
        particle.setColor(_color);
        somethingChanged = true;
    }

    if (_radiusChanged) {
        particle.setRadius(_radius / (float) TREE_SCALE);
        somethingChanged = true;
    }

    if (_velocityChanged) {
        particle.setVelocity(_velocity / (float) TREE_SCALE);
        somethingChanged = true;
    }

    if (_gravityChanged) {
        particle.setGravity(_gravity / (float) TREE_SCALE);
        somethingChanged = true;
    }

    if (_dampingChanged) {
        particle.setDamping(_damping);
        somethingChanged = true;
    }

    if (_lifetimeChanged) {
        particle.setLifetime(_lifetime);
        somethingChanged = true;
    }

    if (_scriptChanged) {
        particle.setScript(_script);
        somethingChanged = true;
    }

    if (_inHandChanged) {
        particle.setInHand(_inHand);
        somethingChanged = true;
    }

    if (_shouldDieChanged) {
        particle.setShouldDie(_shouldDie);
        somethingChanged = true;
    }

    if (_modelURLChanged) {
        particle.setModelURL(_modelURL);
        somethingChanged = true;
    }
    
    if (_modelTranslationChanged) {
        particle.setModelTranslation(_modelTranslation);
        somethingChanged = true;
    }
    
    if (_modelRotationChanged) {
        particle.setModelRotation(_modelRotation);
        somethingChanged = true;
    }
    
    if (_modelScaleChanged) {
        particle.setModelScale(_modelScale);
        somethingChanged = true;
    }
    
    if (somethingChanged) {
        bool wantDebug = false;
        if (wantDebug) {
            uint64_t now = usecTimestampNow();
            int elapsed = now - _lastEdited;
            qDebug() << "ParticleProperties::copyToParticle() AFTER update... edited AGO=" << elapsed <<
                    "now=" << now << " _lastEdited=" << _lastEdited;
        }
        particle.setLastEdited(_lastEdited);
    }
}

void ParticleProperties::copyFromParticle(const Particle& particle) {
    _position = particle.getPosition() * (float) TREE_SCALE;
    _color = particle.getXColor();
    _radius = particle.getRadius() * (float) TREE_SCALE;
    _velocity = particle.getVelocity() * (float) TREE_SCALE;
    _gravity = particle.getGravity() * (float) TREE_SCALE;
    _damping = particle.getDamping();
    _lifetime = particle.getLifetime();
    _script = particle.getScript();
    _inHand = particle.getInHand();
    _shouldDie = particle.getShouldDie();
    _modelURL = particle.getModelURL();
    _modelTranslation = particle.getModelTranslation();
    _modelRotation = particle.getModelRotation();
    _modelScale = particle.getModelScale();

    _id = particle.getID();
    _idSet = true;

    _positionChanged = false;
    _colorChanged = false;
    _radiusChanged = false;
    _velocityChanged = false;
    _gravityChanged = false;
    _dampingChanged = false;
    _lifetimeChanged = false;
    _scriptChanged = false;
    _inHandChanged = false;
    _shouldDieChanged = false;
    _modelURLChanged = false;
    _modelTranslationChanged = false;
    _modelRotationChanged = false;
    _modelScaleChanged = false;
    _defaultSettings = false;
}

QScriptValue ParticlePropertiesToScriptValue(QScriptEngine* engine, const ParticleProperties& properties) {
    return properties.copyToScriptValue(engine);
}

void ParticlePropertiesFromScriptValue(const QScriptValue &object, ParticleProperties& properties) {
    properties.copyFromScriptValue(object);
}


QScriptValue ParticleIDtoScriptValue(QScriptEngine* engine, const ParticleID& id) {
    QScriptValue obj = engine->newObject();
    obj.setProperty("id", id.id);
    obj.setProperty("creatorTokenID", id.creatorTokenID);
    obj.setProperty("isKnownID", id.isKnownID);
    return obj;
}

void ParticleIDfromScriptValue(const QScriptValue &object, ParticleID& id) {
    id.id = object.property("id").toVariant().toUInt();
    id.creatorTokenID = object.property("creatorTokenID").toVariant().toUInt();
    id.isKnownID = object.property("isKnownID").toVariant().toBool();
}



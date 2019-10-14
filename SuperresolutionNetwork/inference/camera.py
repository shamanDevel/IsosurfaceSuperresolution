import math
from enum import Enum

class Orientation(Enum):
    Xp = 1, [1,0,0], [2, -1, -3], True
    Xm = 2, [-1,0,0], [-2, +1, +3], False
    Yp = 3, [0,1,0], [+1, +2, +3], False
    Ym = 4, [0,-1,0], [-1, -2, -3], True
    Zp = 5, [0,0,1], [-3, -1, +2], False
    Zm = 6, [0,0,-1], [+3, +1, -2], True

    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    # ignore the first param since it's already set by __new__
    def __init__(self, _: str, up, permute, invYaw):
        self._up_ = up
        self._permute_ = permute
        self._invYaw_ = invYaw

    def __str__(self):
        return self.value

    # this makes sure that the description is read-only
    @property
    def up(self):
        return self._up_

    @property
    def permute(self):
        return self._permute_

    @property
    def invYaw(self):
        return self._invYaw_

class Camera:
    def __init__(self, resX, resY, origin = [0, 1, -1.7]):
        self.resX = resX
        self.resY = resY
        self.lookAt = [0, 0, 0]
        self.speed = 0.01
        self.zoomspeed = 1.1
        self.orientation = Orientation.Yp

        self.currentDistance, self.currentPitch, self.currentYaw = Camera.toAngles(origin)
        self.baseDistance = self.currentDistance
        self.zoomvalue = 0

    @staticmethod
    def toAngles(pos):
        length = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        pitch = math.asin(pos[1] / length)
        yaw = math.atan2(pos[2], pos[0])
        return length, pitch, yaw
    @staticmethod
    def fromAngles(length, pitch, yaw):
        pos = [0,0,0]
        pos[1] = math.sin(pitch) * length
        pos[0] = math.cos(pitch) * math.cos(yaw) * length
        pos[2] = math.cos(pitch) * math.sin(yaw) * length
        return pos

    def getLookAt(self):
        return self.lookAt

    def getOrigin(self):
        o1 = Camera.fromAngles(self.currentDistance, 
                               self.currentPitch, 
                               self.currentYaw * (-1 if self.orientation.invYaw else +1))
        o2 = [None]*3
        for i in range(3):
            p = self.orientation.permute[i]
            o2[i] = o1[abs(p)-1] * (1 if p>0 else -1)
        return o2

    def getUp(self):
        return self.orientation.up

    def startMove(self):
        self.oldDistance = self.currentDistance
        self.oldPitch = self.currentPitch
        self.oldYaw = self.currentYaw

    def stopMove(self):
        pass

    def move(self, deltax, deltay):
        self.currentPitch = max(math.radians(-80), min(math.radians(80), self.oldPitch + self.speed * deltay))
        self.currentYaw = self.oldYaw + self.speed * deltax
        #print("pitch:", self.currentPitch, ", yaw:", self.currentYaw)

    def zoom(self, delta):
        self.zoomvalue += delta
        self.currentDistance = self.baseDistance * (self.zoomspeed ** self.zoomvalue)
        #print("dist:", self.currentDistance)
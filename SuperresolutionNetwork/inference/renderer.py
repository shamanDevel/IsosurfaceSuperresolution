import subprocess
import numpy as np
from .camera import Camera
import ctypes
import os
import os.path

class Material:
    def __init__(self, iso):
        self.isovalue = iso
        self.diffuseColor = [0.7, 0.2, 0.2]
        self.specularColor = [0.1,0.1,0.1]
        self.specularExponent = 16
        self.light =  'camera'#'-0.3,0.7,-0.5'

class Renderer:
    def __init__(self, renderer, inputfile, material, camera):
        assert isinstance(renderer, str)
        assert isinstance(inputfile, str)
        assert isinstance(material, Material)
        assert isinstance(camera, Camera)
        #launch renderer
        cameraOrigin = camera.getOrigin()
        cameraLookAt = camera.getLookAt()
        cameraUp = camera.getUp()
        args = [
            renderer,
            '-m','iso',
            '--res', '%d,%d'%(camera.resX,camera.resY),
            '--origin', '%5.3f,%5.3f,%5.3f'%(cameraOrigin[0],cameraOrigin[1],cameraOrigin[2]),
            '--lookat', '%5.3f,%5.3f,%5.3f'%(cameraLookAt[0],cameraLookAt[1],cameraLookAt[2]),
            '--up', '%5.3f,%5.3f,%5.3f'%(cameraUp[0],cameraUp[1],cameraUp[2]),
            '--isovalue', str(material.isovalue),
            '--noshading', '0',
            '--diffuse', '%5.3f,%5.3f,%5.3f'%(material.diffuseColor[0],material.diffuseColor[1],material.diffuseColor[2]),
            '--specular', '%5.3f,%5.3f,%5.3f'%(material.specularColor[0],material.specularColor[1],material.specularColor[2]),
            '--exponent', str(material.specularExponent),
            '--light', material.light,
            '--ao', 'world',
            '--aoradius', '0.01',
            inputfile,
            'PIPE'
            ]
        print(' '.join(args))
        self.sp = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=None, stderr=subprocess.PIPE)
        print("Renderer created")
        self.time = 0

    def send_command(self, cmd, value=None):
        #print('Send command "%s"'%cmd.encode("ascii"))
        if value is not None:
            cmd = cmd + "=" + str(value) + "\n"
        self.sp.stdin.write(cmd.encode("ascii"))
        self.sp.stdin.flush()

    def render(self):
        self.send_command("render\n")

    def read_image(self, resX, resY, channels = 12):
        numitems = channels * resY * resX + 1
        imagedata = self.sp.stderr.read(numitems * 4)
        image = np.frombuffer(imagedata, dtype=np.float32, count=numitems)
        self.time = image[-1]
        image = np.reshape(image[0:-1], (channels, resY, resX))
        return image

    def close(self):
        #kill renderer
        self.sp.stdin.write(b"exit\n")
        self.sp.stdin.flush()
        print('exit signal written')
        self.sp.wait(5)

    def get_time(self):
        """Returns the time of the last render pass in seconds"""
        return self.time

class DirectRenderer:
    def __init__(self, renderer):
        assert isinstance(renderer, str)
        assert os.path.exists(renderer)
        os.chdir(os.path.dirname(renderer))
        self.lib = ctypes.cdll.LoadLibrary(os.path.basename(renderer))
        print('Renderer.dll loaded:', self.lib)
        self.time = 0
        # specify types for safety
        self.lib.initGVDB.argtypes = []
        self.lib.initGVDB.restype = ctypes.c_int
        self.lib.loadGrid.argtypes = [ctypes.c_char_p]
        self.lib.loadGrid.restype = ctypes.c_int
        self.lib.setParameter.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.setParameter.restype = ctypes.c_int
        self.lib.render.argtypes = [ctypes.c_ulonglong]
        self.lib.render.restype = ctypes.c_float
        # init
        self.lib.initGVDB()

    def load(self, filename : str):
        self.lib.loadGrid(ctypes.c_char_p(filename.encode("ascii")))

    def send_command(self, cmd, value):
        assert isinstance(cmd, str)
        assert isinstance(value, str)
        self.lib.setParameter(ctypes.c_char_p(cmd.encode("ascii")), 
                              ctypes.c_char_p(value.encode("ascii")))

    def render_direct(self, tensor):
        time = self.lib.render(ctypes.c_ulonglong(tensor.data_ptr()))
        self.time = float(time)
        return self.time

    def get_time(self):
        """Returns the time of the last render pass in seconds"""
        return self.time

    def close(self):
        pass # No-op

class DirectVolumeRenderer:
    def __init__(self, renderer):
        assert isinstance(renderer, str)
        self.lib = ctypes.cdll.LoadLibrary(renderer)
        print('Renderer.dll loaded:', self.lib)
        self.time = 0
        # specify types for safety
        self.lib.init.argtypes = []
        self.lib.cleanup.argtypes = []
        self.lib.loadGrid.argtypes = [ctypes.c_char_p]
        self.lib.loadGrid.restype = ctypes.c_int
        self.lib.setParameter.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.setParameter.restype = ctypes.c_int
        self.c_float_p = ctypes.POINTER(ctypes.c_float)
        self.lib.setTransferFunction.argtypes = [ctypes.c_int, self.c_float_p]
        self.lib.render.argtypes = [ctypes.c_ulonglong]
        self.lib.render.restype = ctypes.c_float
        # init
        self.lib.init()

    def load(self, filename : str):
        self.lib.loadGrid(ctypes.c_char_p(filename.encode("ascii")))

    def send_command(self, cmd, value):
        assert isinstance(cmd, str)
        assert isinstance(value, str)
        self.lib.setParameter(ctypes.c_char_p(cmd.encode("ascii")), 
                              ctypes.c_char_p(value.encode("ascii")))

    def send_transfer_function(self, tf):
        res, c = tf.shape
        assert c==4
        assert tf.dtype == np.float32
        data_p = tf.ctypes.data_as(self.c_float_p)
        self.lib.setTransferFunction(res, data_p)

    def render_direct(self, tensor):
        time = self.lib.render(ctypes.c_ulonglong(tensor.data_ptr()))
        self.time = float(time)
        return self.time

    def get_time(self):
        """Returns the time of the last render pass in seconds"""
        return self.time

    def close(self):
        self.lib.cleanup()
        print("Attempt to close the renderer")
        libHandle = self.lib._handle
        del self.lib
        ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(libHandle))
        self.lib = None
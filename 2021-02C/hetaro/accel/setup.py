from distutils.core import setup, Extension

module1 = Extension('ricochet_hetaro_accel',
    sources=['accel.cpp']
    )

setup (
  name = 'ricochet_hetaro_accel',
  version='1.0',
  description='Fast implementation of Ricochet in C++ for use with Hetaro agents',
  ext_modules=[module1]
  )

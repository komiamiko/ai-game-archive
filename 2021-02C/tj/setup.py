from distutils.core import setup, Extension

module1 = Extension('tj',
    sources=['tj_system.cpp'],
    define_macros=[('_RELEASE', '1')]
    )

setup (
  name = 'tj',
  version='1.0',
  description='tj\'s bot',
  ext_modules=[module1]
  )

cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(UltrasoundAI)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/UltrasoundAI")
set(EXTENSION_CATEGORY "")
set(EXTENSION_CONTRIBUTORS "")
set(EXTENSION_DESCRIPTION "Extension containing modules to deploy deep learning models for segmentation of ultrasound images.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/UltrasoundAI.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/UltrasoundAI/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BreastLesionSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

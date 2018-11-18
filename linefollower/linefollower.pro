TEMPLATE = app
TARGET = linefollower
INCLUDEPATH += .
INCLUDEPATH += /usr/local/include/enki

# Input
HEADERS = Racer.h
SOURCES += Linefollower.cpp Racer.cpp

QT += opengl widgets
CONFIG          += qt warn_on debug
QMAKE_CXXFLAGS += -std=c++0x -march=native -Og

LIBS	+= /usr/local/lib/libenki.a
LIBS	+= -lfcl
LIBS	+= -liir

AMREX_HOME ?= ../../../../amrex
ADV_DIR    = ../

TOP := $(ADV_DIR)

DEBUG	= FALSE
#DEBUG	= TRUE
PRECISION  = DOUBLE
PROFILE    = FALSE

DIM	= 3

COMP    = gnu

USE_MPI   = TRUE
USE_OMP   = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

Bdirs 	:= source source/Kernel
Bpack	+= $(foreach dir, $(Bdirs), $(TOP)/$(dir)/Make.package)
Blocs   += $(foreach dir, $(Bdirs), $(TOP)/$(dir))

include $(Bpack)

INCLUDE_LOCATIONS += $(Blocs)
VPATH_LOCATIONS   += $(Blocs)

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package

all: $(executable) 
	@echo SUCCESS
	
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
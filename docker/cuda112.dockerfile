FROM zxdu20/glm-cuda112
##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# SwissArmyTransformer
##############################################################################
COPY SwissArmyTransformer ${STAGE_DIR}/SwissArmyTransformer
RUN cd ${STAGE_DIR}/SwissArmyTransformer && pip install .

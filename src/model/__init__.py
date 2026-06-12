from model.bsarec import BSARecModel            # teacher
from model.kd_student_v3 import KDStudentV3Model  # proposed student (FreqMamba)
from model.kd_student_v4 import KDStudentV4Model  # v3 + bipolar recency gate

MODEL_DICT = {
    "bsarec": BSARecModel,
    "kdstudent_v3": KDStudentV3Model,
    "kdstudent_v4": KDStudentV4Model,
}

# Older baselines and student variants (mlp/sigma/gru4rec/lrurec/fmlprec/duorec,
# KDStudent v1/v2) were moved to ../../archive/models/. To use them again, move
# the file back and re-add the import + MODEL_DICT entry.

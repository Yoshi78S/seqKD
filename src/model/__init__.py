from model.mlp_student import MLPStudentModel
from model.sigma import SIGMAModel
from model.bsarec import BSARecModel
from model.duorec import DuoRecModel
from model.gru4rec import GRU4RecModel
from model.lrurec import LRURecModel
from model.fmlprec import FMLPRecModel
from model.kd_student import KDStudentModel
from model.kd_student_v2 import KDStudentV2Model
from model.kd_student_v3 import KDStudentV3Model

MODEL_DICT = {
    "mlp": MLPStudentModel,
    "mlp_student": MLPStudentModel,
    "sigma": SIGMAModel,
    "bsarec": BSARecModel,
    "duorec": DuoRecModel,
    "gru4rec": GRU4RecModel,
    "lrurec": LRURecModel,
    "fmlprec": FMLPRecModel,
    "kdstudent": KDStudentModel,
    "kdstudent_v2": KDStudentV2Model,
    "kdstudent_v3": KDStudentV3Model,
}

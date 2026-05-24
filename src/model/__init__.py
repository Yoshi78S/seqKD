from model.mlp_student import MLPStudentModel
from model.sigma import SIGMAModel
from model.bsarec import BSARecModel
from model.duorec import DuoRecModel
from model.gru4rec import GRU4RecModel
from model.lrurec import LRURecModel
from model.fmlprec import FMLPRecModel
from model.kd_student import KDStudentModel

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
}

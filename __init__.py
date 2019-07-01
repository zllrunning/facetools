from util.extract_feature import extract_feature
from backbone.model_irse import IR_50
from align.detector import detect_faces
from align.visualization_utils import show_results
from align.face_align import align
from parsing.face_parsing import parsing, vis_parsing_maps
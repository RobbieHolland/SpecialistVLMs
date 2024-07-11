from evaluation.tasks.specialist.biomarker_detection import SpecialistDetection
from evaluation.tasks.specialist.other import SpecialistOther
from evaluation.tasks.specialist.referral import SpecialistReferral

class SpecialistTasks:
    other_tasks = SpecialistOther
    detection_tasks = SpecialistDetection
    referral_tasks = SpecialistReferral
    
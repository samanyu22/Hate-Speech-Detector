import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.models.transformer_classifier import MultilingualTransformerEngine

raw_data = """id531 | People belonging to that religion should not be allowed to adopt children. | hateful
id532 | The adoption agency has updated its eligibility guidelines. | unhateful
id533 | Employees from that ethnic background should be denied access to company training programmes. | hateful
id534 | The company has introduced a new professional development programme for employees. | unhateful
id535 | People of that caste should not be permitted to participate in village decision-making. | hateful
id536 | The village council is collecting suggestions about its development projects. | unhateful
id537 | Students from that nationality should not be allowed to receive university research grants. | hateful
id538 | The university has announced a new research grant application process. | unhateful
id539 | People with that disability should not be allowed to participate in professional examinations. | hateful
id540 | The examination board has published guidelines for candidates requiring additional arrangements. | unhateful
id541 | People from that religious community should be refused membership in the residents' association. | hateful
id542 | The residents' association is organising its annual general meeting. | unhateful
id543 | People of that ethnic group should not be allowed to register as professional athletes. | hateful
id544 | The sports authority has introduced a revised athlete registration procedure. | unhateful
id545 | Women from that community should not be allowed to inherit agricultural land. | hateful
id546 | The agriculture department is publishing information about land registration. | unhateful
id547 | People from that country should be excluded from international research collaborations. | hateful
id548 | The research institute is inviting international universities to collaborate. | unhateful
id549 | People belonging to that caste should not be allowed to operate public transport vehicles. | hateful
id550 | The transport department is conducting a review of vehicle licensing rules. | unhateful
id551 | People from that minority religion should be refused access to legal aid programmes. | hateful
id552 | The legal aid organisation has opened a new consultation office. | unhateful
id553 | People of that ethnic background should not be allowed to become university professors. | hateful
id554 | The university has announced vacancies in its academic departments. | unhateful
id555 | People with visual disabilities should not be permitted to access online banking services. | hateful
id556 | The bank is improving the accessibility of its online banking platform. | unhateful
id557 | People from that nationality should not be allowed to participate in scientific conferences. | hateful
id558 | The conference committee has published its registration requirements. | unhateful
id559 | Members of that religious group should be excluded from public insurance programmes. | hateful
id560 | The insurance department has released information about its latest public scheme. | unhateful
id561 | People of that gender should not be allowed to apply for military scholarships. | hateful
id562 | The scholarship office has announced its annual application deadline. | unhateful
id563 | People from that caste should not be permitted to serve as elected representatives. | hateful
id564 | The election commission has released information about candidate nominations. | unhateful
id565 | People from that ethnic community should not be allowed to establish educational institutions. | hateful
id566 | The education board is reviewing applications for new schools. | unhateful
id567 | People with hearing impairments should be prohibited from attending court hearings. | hateful
id568 | The court is reviewing its arrangements for accessible hearings. | unhateful
id569 | People of that religion should be denied access to public scholarship examinations. | hateful
id570 | The examination authority has released the scholarship examination timetable. | unhateful
id571 | People from that country should not be permitted to participate in international student exchanges. | hateful
id572 | The international office is preparing information about student exchange opportunities. | unhateful
id573 | People of that caste should not be allowed to become members of cooperative societies. | hateful
id574 | The cooperative society is accepting applications for new memberships. | unhateful
id575 | People with neurological disabilities should not be allowed to apply for research fellowships. | hateful
id576 | The fellowship programme has published its research proposal requirements. | unhateful
id577 | People from that ethnic group should be refused entry to private sports academies. | hateful
id578 | The sports academy is accepting registrations for its summer programme. | unhateful
id579 | People of that religious background should not be allowed to become registered nurses. | hateful
id580 | The nursing council has published its professional registration guidelines. | unhateful
id581 | Women should be excluded from negotiations involving business contracts. | hateful
id582 | The business association is conducting a workshop on contract negotiations. | unhateful
id583 | Men should not be permitted to work in early childhood education centres. | hateful
id584 | The education centre is recruiting qualified early childhood educators. | unhateful
id585 | People who identify as non-binary should not be allowed to access student accommodation. | hateful
id586 | The university accommodation office has published its room allocation policy. | unhateful
id587 | Transgender people should be denied access to professional licensing examinations. | hateful
id588 | The licensing authority has announced the dates of its upcoming examinations. | unhateful
id589 | People of a particular sexual orientation should not be allowed to join a research team. | hateful
id590 | The research team is inviting applications from qualified candidates. | unhateful
id591 | People with dwarfism should not be permitted to enter public swimming competitions. | hateful
id592 | The swimming federation has published its competition registration rules. | unhateful
id593 | People with Down syndrome should be excluded from vocational certification programmes. | hateful
id594 | The vocational institute has added several new certification courses. | unhateful
id595 | People with speech disabilities should not be allowed to apply for public speaking awards. | hateful
id596 | The organisers have announced the judging criteria for the public speaking competition. | unhateful
id597 | People with epilepsy should not be allowed to access university laboratories. | hateful
id598 | The university laboratory has introduced updated safety procedures. | unhateful
id599 | People with mobility disabilities should be denied access to public examination centres. | hateful
id600 | The examination authority is mapping accessible examination centres. | unhateful
id601 | People from that ethnic group should not be allowed to serve on hospital management boards. | hateful
id602 | The hospital is inviting applications for its administrative advisory committee. | unhateful
id603 | Members of that religious community should be denied access to public childcare facilities. | hateful
id604 | The childcare department has announced new registration dates. | unhateful
id605 | People from that nationality should not be allowed to receive professional awards. | hateful
id606 | The professional association has announced the nominations for its annual awards. | unhateful
id607 | People from that caste should not be allowed to apply for agricultural subsidies. | hateful
id608 | The agriculture office has published details of its subsidy application process. | unhateful
id609 | People from that minority group should not be permitted to participate in public consultations about urban planning. | hateful
id610 | The urban planning department is inviting residents to comment on its development proposal. | unhateful
id611 | People of that religion should not be allowed to access government-funded libraries. | hateful
id612 | The government library has expanded its digital catalogue. | unhateful
id613 | People with intellectual disabilities should not be allowed to participate in community theatre. | hateful
id614 | The theatre group is holding auditions for its next production. | unhateful
id615 | Older people should be excluded from public computer training programmes. | hateful
id616 | The community centre is offering free computer lessons to interested residents. | unhateful
id617 | Young people from that nationality should not be permitted to apply for internships. | hateful
id618 | The organisation has opened applications for its summer internship programme. | unhateful
id619 | People from rural communities should be prohibited from applying for city government positions. | hateful
id620 | The city administration has published its latest recruitment notice. | unhateful
id621 | People who speak a minority language should not be allowed to participate in public broadcasting. | hateful
id622 | The broadcasting service is accepting proposals for regional language programmes. | unhateful
id623 | Members of that cultural community should be refused access to public museums. | hateful
id624 | The museum is introducing guided tours about regional cultural history. | unhateful
id625 | People from that country should not be allowed to receive disaster insurance. | hateful
id626 | The insurance department is reviewing its disaster coverage documentation. | unhateful
id627 | People of that ethnic background should not be permitted to join the national debating team. | hateful
id628 | The debating association has announced its selection schedule. | unhateful
id629 | People belonging to that caste should not be allowed to use public sports equipment. | hateful
id630 | The municipal sports complex has installed new fitness equipment. | unhateful
id631 | People from that religious group should be excluded from public health awareness campaigns. | hateful
id632 | The health department is preparing a campaign about preventive healthcare. | unhateful
id633 | People with autism should not be permitted to participate in university student elections. | hateful
id634 | The student election committee has published its nomination process. | unhateful
id635 | People with visual impairments should be denied access to digital educational resources. | hateful
id636 | The education portal is adding compatibility with screen-reading applications. | unhateful
id637 | People with hearing impairments should not be allowed to participate in online classes. | hateful
id638 | The online learning platform is adding automatic caption support. | unhateful
id639 | People with physical disabilities should not be allowed to participate in scientific fieldwork. | hateful
id640 | The research department is planning its next field study. | unhateful
id641 | People from that ethnic group should not be allowed to become public accountants. | hateful
id642 | The accounting institute has announced its next professional certification examination. | unhateful
id643 | People from that religious background should not be permitted to receive public housing loans. | hateful
id644 | The housing department has updated its home loan information portal. | unhateful
id645 | Women should not be permitted to participate in company board elections. | hateful
id646 | The company has announced the schedule for its board elections. | unhateful
id647 | Men should not be allowed to receive parental leave benefits. | hateful
id648 | The organisation has revised its parental leave policy. | unhateful
id649 | People from that nationality should not be allowed to access public employment exchanges. | hateful
id650 | The employment exchange has launched an online vacancy search service. | unhateful
id651 | People from that caste should be prevented from joining local environmental committees. | hateful
id652 | The environmental committee is organising a neighbourhood clean-up programme. | unhateful
id653 | Members of that ethnic group should not be allowed to participate in public art exhibitions. | hateful
id654 | The art gallery is accepting submissions for its annual exhibition. | unhateful
id655 | People belonging to that minority religion should not be allowed to use public libraries after sunset. | hateful
id656 | The library has extended its evening opening hours during examination season. | unhateful
id657 | People with disabilities should not be allowed to access online government forms. | hateful
id658 | The government portal is simplifying its online application forms. | unhateful
id659 | People from that country should be denied access to public legal information services. | hateful
id660 | The legal information centre has launched a new website for residents. | unhateful
id661 | People from that ethnic background should not be allowed to become school principals. | hateful
id662 | The education department has published the eligibility criteria for school principals. | unhateful
id663 | People of that caste should not be permitted to receive awards for academic achievement. | hateful
id664 | The university is organising an awards ceremony for outstanding students. | unhateful
id665 | People of a particular gender should not be allowed to participate in professional engineering associations. | hateful
id666 | The engineering association is hosting its annual technical conference. | unhateful
id667 | People of that sexual orientation should not be allowed to receive public arts funding. | hateful
id668 | The arts council has announced its funding application schedule. | unhateful
id669 | People with learning disabilities should not be allowed to use public examination libraries. | hateful
id670 | The examination library has added new reference materials. | unhateful
id671 | People from that region should be refused membership in national professional organisations. | hateful
id672 | The professional organisation has opened its membership renewal portal. | unhateful
id673 | People from that community should not be allowed to participate in local heritage projects. | hateful
id674 | The heritage department is documenting historic buildings in the region. | unhateful
id675 | People from a particular nationality should not be permitted to work in public research laboratories. | hateful
id676 | The research laboratory has announced its latest recruitment opportunities. | unhateful
id677 | People from that religious community should be excluded from public agricultural programmes. | hateful
id678 | The agriculture department is organising a workshop about crop management. | unhateful
id679 | People from that caste should not be allowed to register for public skill development courses. | hateful
id680 | The skill development centre has announced its next training batch. | unhateful
id681 | People with a particular disability should be denied access to public information hotlines. | hateful
id682 | The public information service has introduced additional communication channels. | unhateful
id683 | People of that ethnic group should not be allowed to receive community leadership awards. | hateful
id684 | The community organisation has announced nominations for its annual awards. | unhateful
id685 | Women from that nationality should not be allowed to participate in international academic exchanges. | hateful
id686 | The academic exchange office has published its application instructions. | unhateful
id687 | Men from that ethnic community should not be permitted to work in public childcare services. | hateful
id688 | The childcare department is organising a workshop for its employees. | unhateful
id689 | People who follow that minority religion should not be allowed to access public sports scholarships. | hateful
id690 | The sports scholarship programme has published its selection criteria. | unhateful
id691 | People from that caste should not be allowed to receive community development grants. | hateful
id692 | The development office is accepting proposals for community improvement projects. | unhateful
id693 | People with hearing impairments should be prohibited from accessing online public meetings. | hateful
id694 | The council is introducing live captions for its online meetings. | unhateful
id695 | People from that ethnic group should not be allowed to participate in university graduation ceremonies. | hateful
id696 | The university has published the schedule for its graduation ceremony. | unhateful
id697 | People from that nationality should be denied access to public career counselling. | hateful
id698 | The career centre has opened registrations for its counselling sessions. | unhateful
id699 | People from that religious community should not be allowed to participate in public science exhibitions. | hateful
id700 | The science department is organising an exhibition for local schools. | unhateful
id701 | People of that caste should not be permitted to join public library committees. | hateful
id702 | The library committee is reviewing its book acquisition budget. | unhateful
id703 | People from that ethnic background should be denied access to professional mentoring programmes. | hateful
id704 | The mentoring programme is accepting applications from early-career professionals. | unhateful
id705 | People with physical disabilities should not be allowed to participate in public photography competitions. | hateful
id706 | The photography club has announced the theme of its next competition. | unhateful
id707 | People from that country should be excluded from public entrepreneurship programmes. | hateful
id708 | The entrepreneurship centre is organising a workshop for new business owners. | unhateful
id709 | People of that religion should not be permitted to access public cultural grants. | hateful
id710 | The cultural department has announced a new grant application window. | unhateful
id711 | People from that caste should not be allowed to serve on school parent committees. | hateful
id712 | The school is inviting parents and guardians to its next committee meeting. | unhateful
id713 | People with visual disabilities should be excluded from public technology training. | hateful
id714 | The technology centre has introduced accessible computer training sessions. | unhateful
id715 | People of that gender identity should not be permitted to access public career fairs. | hateful
id716 | The career fair will feature employers from several industries. | unhateful
id717 | People from that ethnic group should be denied access to public innovation grants. | hateful
id718 | The innovation department has opened applications for its latest grant programme. | unhateful
id719 | People from that nationality should not be allowed to participate in public environmental research. | hateful
id720 | The environmental institute is recruiting volunteers for a biodiversity study. | unhateful
id721 | People belonging to that religious group should not be permitted to join public education committees. | hateful
id722 | The education committee is reviewing the school's new curriculum proposal. | unhateful
id723 | People from that caste should not be allowed to participate in public water management programmes. | hateful
id724 | The water authority is collecting feedback about its conservation programme. | unhateful
id725 | People with intellectual disabilities should be denied access to public vocational counselling. | hateful
id726 | The vocational centre provides career guidance to people exploring different professions. | unhateful
id727 | People from that ethnic community should not be allowed to participate in public heritage preservation work. | hateful
id728 | The heritage organisation is recruiting volunteers to document historic monuments. | unhateful
id729 | People from that religion should be prohibited from receiving public educational awards. | hateful
id730 | The education board has announced the winners of its annual academic awards. | unhateful"""

lines = [line.strip() for line in raw_data.strip().split("\n") if line.strip()]
print(f"Total lines parsed: {len(lines)}")

samples = []
for line in lines:
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 3:
        samples.append({
            "id": parts[0],
            "text": parts[1],
            "label": parts[2].lower()
        })
    else:
        print("Malformed line:", line)

print(f"Total valid samples: {len(samples)}")

clf = MultilingualTransformerEngine()
clf.load_checkpoint()

y_true = []
y_pred = []
results = []

for s in samples:
    gold_str = s["label"]
    gold_label = 1 if gold_str == "hateful" else 0
    
    pred_res = clf.predict(s["text"])
    pred_label = pred_res["label"]
    pred_name = "hateful" if pred_label == 1 else "unhateful"
    confidence = pred_res["confidence"]
    raw_h_prob = pred_res["raw_hateful_probability"]
    
    y_true.append(gold_label)
    y_pred.append(pred_label)
    
    is_match = (gold_label == pred_label)
    results.append({
        "id": s["id"],
        "text": s["text"],
        "gold_label": gold_str,
        "gold_numeric": gold_label,
        "pred_label": pred_name,
        "pred_numeric": pred_label,
        "confidence": confidence,
        "prob_hateful": raw_h_prob,
        "match": is_match
    })

acc = accuracy_score(y_true, y_pred)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

mismatches = [r for r in results if not r["match"]]

eval_summary = {
    "total_samples": len(samples),
    "accuracy": round(float(acc), 4),
    "macro_f1": round(float(f1_m), 4),
    "macro_precision": round(float(prec_m), 4),
    "macro_recall": round(float(rec_m), 4),
    "confusion_matrix": {
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp)
    },
    "per_class": {
        "unhateful": {
            "precision": round(float(prec_cls[0]), 4),
            "recall": round(float(rec_cls[0]), 4),
            "f1": round(float(f1_cls[0]), 4),
            "support": int(support[0])
        },
        "hateful": {
            "precision": round(float(prec_cls[1]), 4),
            "recall": round(float(rec_cls[1]), 4),
            "f1": round(float(f1_cls[1]), 4),
            "support": int(support[1])
        }
    },
    "mismatches": mismatches,
    "all_results": results
}

with open("scratch/eval_batch3_results.json", "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, indent=2)

print("\n" + "="*50)
print("BATCH 3 EVALUATION COMPLETED (200 SAMPLES):")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Macro F1: {f1_m:.4f}")
print(f"Macro Precision: {prec_m:.4f} | Macro Recall: {rec_m:.4f}")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Total Mismatches: {len(mismatches)}")
print("="*50)


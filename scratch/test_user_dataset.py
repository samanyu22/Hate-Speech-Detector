import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import csv
import io
import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.models.transformer_classifier import MultilingualTransformerEngine

raw_tsv = """ID	Text	Label
id1	People from that religion are inferior and should not be treated as equals.	hateful
id2	People from many religions live peacefully in our city.	unhateful
id3	Members of that faith do not deserve the same rights as everyone else.	hateful
id4	Everyone should have equal rights regardless of their faith.	unhateful
id5	That religious community should be excluded from our neighbourhood.	hateful
id6	I disagree with that religious institution's policy, not with its followers.	unhateful
id7	People of that religion are all dishonest and cannot be trusted.	hateful
id8	Religious leaders should be accountable for their public statements.	unhateful
id9	Nobody from that faith should be allowed to work in public service.	hateful
id10	People should be free to follow a religion or no religion.	unhateful
id11	That caste is naturally less intelligent than other castes.	hateful
id12	Our school welcomes students from every religious background.	unhateful
id13	People born into that caste should stay away from our community.	hateful
id14	That caste system is unfair and should be challenged.	unhateful
id15	That caste should never be allowed to lead our organisation.	hateful
id16	No one should face discrimination because of their caste.	unhateful
id17	Members of that caste are dirty and should be kept separate.	hateful
id18	People from every caste deserve equal access to education.	unhateful
id19	People from that caste do not deserve equal opportunities.	hateful
id20	We should discuss caste discrimination respectfully and honestly.	unhateful
id21	Women are too weak-minded to make important decisions.	hateful
id22	I disagree with the decision made by the local council.	unhateful
id23	Women should not be allowed to study or work outside the home.	hateful
id24	The government's immigration policy needs improvement.	unhateful
id25	Girls from that community are not fit to become leaders.	hateful
id26	Some immigration procedures are too slow and confusing.	unhateful
id27	Men are the only people capable of doing serious work.	hateful
id28	People should be judged by their actions, not their nationality.	unhateful
id29	People should be rejected from jobs simply because they are women.	hateful
id30	Students from different countries bring useful perspectives to class.	unhateful
id31	People with disabilities are a burden and should be excluded from society.	hateful
id32	That politician's speech was misleading and poorly researched.	unhateful
id33	Disabled people should not be allowed to participate in public life.	hateful
id34	I strongly oppose the party's economic policy.	unhateful
id35	People with that disability are useless to the community.	hateful
id36	The company should improve its hiring process.	unhateful
id37	Nobody should hire a person just because they use a wheelchair.	hateful
id38	That manager treated the staff unfairly.	unhateful
id39	People with intellectual disabilities should not have a voice in society.	hateful
id40	The film's writing was weak and the pacing was poor.	unhateful
id41	Immigrants from that country are parasites who ruin every place they enter.	hateful
id42	That comment was rude, but it does not represent an entire community.	unhateful
id43	People of that nationality should be forced out of our town.	hateful
id44	I dislike the behaviour of that individual.	unhateful
id45	Nobody from that country deserves a fair chance here.	hateful
id46	That person made an irresponsible decision.	unhateful
id47	That nationality is made up of criminals and dishonest people.	hateful
id48	The referee made a terrible call during the match.	unhateful
id49	People from that country should be banned from our schools.	hateful
id50	This software is frustrating and badly designed.	unhateful
id51	People of that ethnicity are naturally violent and dangerous.	hateful
id52	That team's strategy was ineffective today.	unhateful
id53	That ethnic group should be kept out of our city.	hateful
id54	I disagree with the argument in this article.	unhateful
id55	People from that ethnic background are inferior to everyone else.	hateful
id56	The speaker's evidence does not support the conclusion.	unhateful
id57	That ethnic community should not be allowed to own businesses here.	hateful
id58	People should be evaluated based on relevant qualifications.	unhateful
id59	Children from that ethnic group should be kept out of our schools.	hateful
id60	Everyone deserves a fair chance to apply for the job.	unhateful
id61	People should be denied housing because of their religion.	hateful
id62	Accessibility improvements help make public spaces usable for everyone.	unhateful
id63	Do not rent apartments to anyone from that caste.	hateful
id64	Wheelchair users should have access to every public building.	unhateful
id65	Keep people of that ethnicity out of this public event.	hateful
id66	People with disabilities deserve respect and equal opportunities.	unhateful
id67	Reject applicants who belong to that religious community.	hateful
id68	Neurodivergent students should receive appropriate learning support.	unhateful
id69	People from that nationality should not be allowed to vote.	hateful
id70	People should not be judged by their gender.	unhateful
id71	That community is disgusting because of who they are.	hateful
id72	Women and men should receive equal pay for equal work.	unhateful
id73	People like them are not real citizens and never will be.	hateful
id74	Girls should have the same opportunities in science and technology.	unhateful
id75	Our town would be better if that ethnic group disappeared from public life.	hateful
id76	Men can be caring parents and excellent teachers.	unhateful
id77	That religious minority does not belong in this country.	hateful
id78	Gender should not determine someone's career options.	unhateful
id79	People from that caste should know their place and stay beneath us.	hateful
id80	People should be free to express their gender identity safely.	unhateful
id81	All people of that nationality are stupid and incapable.	hateful
id82	Everyone deserves dignity regardless of sexual orientation.	unhateful
id83	That ethnic group is naturally dishonest, so avoid every one of them.	hateful
id84	Schools should prevent bullying based on identity.	unhateful
id85	People of that religion are a threat just by existing here.	hateful
id86	People from different ethnic backgrounds can work well together.	unhateful
id87	That community should have fewer rights than the rest of us.	hateful
id88	Our neighbourhood benefits from welcoming newcomers.	unhateful
id89	Do not let anyone from that ethnic group join our club.	hateful
id90	Citizens should debate national policies without attacking people by origin.	unhateful
id91	People from that country should be treated as second-class residents.	hateful
id92	I disagree with that country's government, not its citizens.	unhateful
id93	That caste makes people unclean, so keep them away.	hateful
id94	The government should explain how it handles asylum applications.	unhateful
id95	Women from that background should never be trusted with authority.	hateful
id96	Some tourists ignored local rules, and the authorities should address it.	unhateful
id97	Men from that ethnic group are all born criminals.	hateful
id98	Criticising a government is not the same as attacking a nationality.	unhateful
id99	People of that religion should be removed from our school.	hateful
id100	That religious practice is unfamiliar to me, so I want to learn about it.	unhateful
id101	That minority group is ruining society simply by being here.	hateful
id102	I do not share that belief, but others have a right to hold it.	unhateful
id103	People from that community should not be allowed to live in this area.	hateful
id104	People should be allowed to peacefully criticise religious ideas.	unhateful
id105	That nationality is inferior and should remain subordinate.	hateful
id106	Historical caste discrimination should be studied in schools.	unhateful
id107	People with hearing impairments cannot contribute anything useful.	hateful
id108	The article describes discrimination against a minority community.	unhateful
id109	People with visual impairments should not be included in our team.	hateful
id110	We should support people who experience racism.	unhateful
id111	Autistic people are defective and should be avoided.	hateful
id112	That post is an example of hateful rhetoric and should be reviewed.	unhateful
id113	People with mental health conditions are unfit to be part of society.	hateful
id114	I am quoting the harmful statement to explain why it is discriminatory.	unhateful
id115	Older people are worthless and should be pushed out of public life.	hateful
id116	Do not generalise one person's behaviour to an entire group.	unhateful
id117	Young people from that ethnic group should not be given opportunities.	hateful
id118	People from every background should be treated fairly.	unhateful
id119	People should be excluded from sports because of their gender identity.	hateful
id120	Our club welcomes members regardless of disability.	unhateful
id121	Nobody from that sexual orientation should be welcome in our club.	hateful
id122	The event venue needs ramps and accessible restrooms.	unhateful
id123	People of that identity do not deserve respect or equal treatment.	hateful
id124	Older adults have valuable experience and should be included.	unhateful
id125	That group should be denied access to public facilities.	hateful
id126	Young people deserve to be heard in community decisions.	unhateful
id127	People should not be served at this shop because of their religion.	hateful
id128	People should not lose opportunities because of their age.	unhateful
id129	Do not invite anyone from that caste to the celebration.	hateful
id130	That employee missed the deadline and needs to improve.	unhateful
id131	People from that ethnic group should not be allowed in this apartment complex.	hateful
id132	The customer was disrespectful to the cashier.	unhateful
id133	That nationality should be treated as unwelcome wherever they go.	hateful
id134	I think the actor gave a poor performance.	unhateful
id135	Members of that faith are less human than the rest of us.	hateful
id136	The restaurant's service was slow and the food was cold.	unhateful
id137	That community should be removed from our society.	hateful
id138	The app keeps crashing after the latest update.	unhateful
id139	People of that race should never hold positions of power.	hateful
id140	The policy is confusing and should be rewritten.	unhateful
id141	That race is naturally less capable and should accept fewer opportunities.	hateful
id142	The city should invest more in public transport.	unhateful
id143	People from that background should not be allowed to represent our school.	hateful
id144	The school should review its attendance rules.	unhateful
id145	People of that caste are born to serve others and should not expect equality.	hateful
id146	The debate focused on ideas rather than personal identities.	unhateful
id147	That religion makes every person who follows it inferior.	hateful
id148	That joke was offensive, and I asked the speaker to stop.	unhateful
id149	People from that country should be kept out because they are all worthless.	hateful
id150	Please criticise the proposal without insulting its supporters.	unhateful
id151	Women are naturally unfit for technical careers.	hateful
id152	We should challenge stereotypes when we hear them.	unhateful
id153	Men from that community should never be allowed to teach children.	hateful
id154	People should not be excluded from housing because of identity.	unhateful
id155	People with mobility disabilities should be kept out of public buildings.	hateful
id156	Equal treatment should apply to everyone in the workplace.	unhateful
id157	People with speech disabilities have nothing worth hearing.	hateful
id158	The university should investigate the reported discrimination.	unhateful
id159	People of that ethnicity should not be allowed to become neighbours.	hateful
id160	Support services should be available to students who need them.	unhateful
id161	That minority should be denied the right to participate in elections.	hateful
id162	People who use sign language should have access to interpreters.	unhateful
id163	People from that caste should not be allowed to enter this temple.	hateful
id164	Blind and low-vision visitors should be able to navigate the museum.	unhateful
id165	People from that religion should not be allowed to enter this place of worship.	hateful
id166	Autistic people have different strengths and support needs.	unhateful
id167	Do not employ anyone who belongs to that ethnic group.	hateful
id168	Mental health challenges do not make someone less worthy of respect.	unhateful
id169	People of that nationality should be denied access to healthcare.	hateful
id170	People should be assessed individually rather than by stereotypes.	unhateful
id171	That gender is too irrational to make decisions.	hateful
id172	That candidate's plan lacks detail on education funding.	unhateful
id173	People who are transgender should be excluded from ordinary public life.	hateful
id174	The mayor's decision has had negative effects on local businesses.	unhateful
id175	People of that sexual orientation are not fit to be part of our community.	hateful
id176	I oppose the law because it restricts freedom of expression.	unhateful
id177	People from that ethnic group are inherently inferior.	hateful
id178	The news report needs stronger evidence and more context.	unhateful
id179	That religious group should be made to feel unwelcome everywhere.	hateful
id180	That historical leader made decisions that deserve criticism.	unhateful
id181	People of that caste should be barred from leadership positions.	hateful
id182	The organisation should publish its spending records.	unhateful
id183	People from that country should never be accepted as our equals.	hateful
id184	The company should be criticised for unsafe working conditions.	unhateful
id185	People with disabilities should be hidden away from public view.	hateful
id186	People can disagree strongly without targeting someone's identity.	unhateful
id187	That community deserves fewer rights because of its identity.	hateful
id188	Let's keep the discussion focused on the issue.	unhateful
id189	People of that race should not be allowed to attend this university.	hateful
id190	That statement stereotypes an entire group and should be corrected.	unhateful
id191	People from that religion are all dangerous and should be avoided.	hateful
id192	People of all faiths should be safe at school.	unhateful
id193	People from that nationality should not be allowed to become citizens.	hateful
id194	Everyone should be able to access healthcare without discrimination.	unhateful
id195	That ethnic group should be segregated from everyone else.	hateful
id196	Public services should be available to residents fairly.	unhateful
id197	People of that caste should not be allowed to share public spaces.	hateful
id198	People should be hired based on skills and experience.	unhateful
id199	Women from that religion should not be allowed to make their own choices.	hateful
id200	An individual's criminal behaviour does not define their whole nationality.	unhateful
id201	People with learning disabilities should not be allowed to study with others.	hateful
id202	One person's actions should not be blamed on an entire religion.	unhateful
id203	People from that community are inherently untrustworthy.	hateful
id204	The post criticises a political movement, not an ethnic group.	unhateful
id205	That group should be denied equal protection under the law.	hateful
id206	The speaker is discussing a policy and not insulting its supporters.	unhateful
id207	People should be ashamed of belonging to that ethnicity.	hateful
id208	The complaint is about poor service, not the employee's identity.	unhateful
id209	People from that community should not be allowed to run for public office.	hateful
id210	That comment attacks an idea rather than a protected group.	unhateful"""

reader = csv.DictReader(io.StringIO(raw_tsv.strip()), delimiter="\t")
data = list(reader)

clf = MultilingualTransformerEngine()
clf.load_checkpoint()

results = []
y_true = []
y_pred = []

for row in data:
    sample_id = row["ID"]
    text = row["Text"]
    gold_str = row["Label"].strip().lower()
    gold_label = 1 if gold_str == "hateful" else 0
    
    pred_res = clf.predict(text)
    pred_label = pred_res["label"]
    pred_name = "hateful" if pred_label == 1 else "unhateful"
    confidence = pred_res["confidence"]
    raw_h_prob = pred_res["raw_hateful_probability"]
    
    y_true.append(gold_label)
    y_pred.append(pred_label)
    
    is_match = (gold_label == pred_label)
    results.append({
        "id": sample_id,
        "text": text,
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

summary = {
    "total_samples": len(data),
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
    "mismatches": [r for r in results if not r["match"]],
    "all_results": results
}

with open("scratch/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Evaluation Complete!")
print(f"Total: {len(data)} | Accuracy: {acc*100:.2f}% | Macro F1: {f1_m:.4f}")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"Total Mismatches: {len(summary['mismatches'])}")

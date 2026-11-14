import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.models.transformer_classifier import MultilingualTransformerEngine

raw_data = """id211 | Sikh applicants should be rejected solely because of their religion. | hateful
id212 | The public library has introduced new books about Sikh history. | unhateful
id213 | Christians from that community should be denied access to local schools. | hateful
id214 | The school is organising an exhibition about Christian art and architecture. | unhateful
id215 | Muslims should never be allowed to hold senior positions in this company. | hateful
id216 | The local council is reviewing its policy on religious holiday allowances. | unhateful
id217 | Jewish families should not be permitted to purchase homes in this neighbourhood. | hateful
id218 | The museum is hosting a lecture about Jewish history. | unhateful
id219 | Hindus from that region should be refused admission to our university. | hateful
id220 | The university is conducting research on Hindu philosophical traditions. | unhateful
id221 | Buddhists should have fewer legal protections than other citizens. | hateful
id222 | The documentary examines the history of Buddhist monasteries. | unhateful
id223 | Atheists should be excluded from all public leadership roles. | hateful
id224 | People can have different beliefs about the existence of God. | unhateful
id225 | Jains should not be allowed to operate businesses in this district. | hateful
id226 | The exhibition explains the principles of Jain philosophy. | unhateful
id227 | Refugees from that country should automatically be denied education. | hateful
id228 | The government has announced a new scholarship for refugee students. | unhateful
id229 | People of African descent should not be allowed to join this association. | hateful
id230 | The association is celebrating the contributions of African writers. | unhateful
id231 | Roma families should be refused access to public housing. | hateful
id232 | The report examines housing conditions among Roma communities. | unhateful
id233 | Indigenous people should not be allowed to own land. | hateful
id234 | Indigenous communities are campaigning for recognition of their land rights. | unhateful
id235 | Adivasi students should be prevented from applying to this college. | hateful
id236 | The college has introduced scholarships for Adivasi students. | unhateful
id237 | Dalit employees should never be promoted to management positions. | hateful
id238 | The organisation is investigating allegations of caste discrimination. | unhateful
id239 | People from the Brahmin community should be denied access to public services. | hateful
id240 | The article discusses the historical development of caste classifications. | unhateful
id241 | Interfaith couples should be prohibited from renting apartments here. | hateful
id242 | The housing authority has published new guidelines against discrimination. | unhateful
id243 | People who speak Urdu should not be permitted to attend this school. | hateful
id244 | The language department has added Urdu to its optional courses. | unhateful
id245 | Tamil-speaking residents should be denied government assistance. | hateful
id246 | The city is expanding its Tamil-language public information services. | unhateful
id247 | Bengali families should not be allowed to settle in this neighbourhood. | hateful
id248 | The cultural festival will feature Bengali literature and music. | unhateful
id249 | People from Northeast India should be barred from university hostels. | hateful
id250 | The university has opened a cultural centre for students from Northeast India. | unhateful
id251 | People of a particular skin colour should receive lower salaries. | hateful
id252 | The company is reviewing its salary structure for all employees. | unhateful
id253 | Applicants with darker skin should automatically fail the interview. | hateful
id254 | The recruitment team has introduced a standardised interview process. | unhateful
id255 | People with albinism should be excluded from public competitions. | hateful
id256 | The awareness campaign explains the needs of people with albinism. | unhateful
id257 | People with dwarfism should be refused entry to this amusement park. | hateful
id258 | The amusement park is reviewing accessibility for visitors with different needs. | unhateful
id259 | People with Down syndrome should not be allowed to attend ordinary schools. | hateful
id260 | The school provides individual learning support to students with Down syndrome. | unhateful
id261 | Deaf applicants should never be considered for customer-facing jobs. | hateful
id262 | The company has introduced sign-language interpretation during interviews. | unhateful
id263 | People who stutter should be prohibited from participating in debates. | hateful
id264 | The debate club offers additional preparation time to every participant who needs it. | unhateful
id265 | People with epilepsy should be excluded from university activities. | hateful
id266 | The university is reviewing its emergency support arrangements for students with epilepsy. | unhateful
id267 | Blind employees should not be permitted to work in this office. | hateful
id268 | The office has installed screen-reading software for employees who need it. | unhateful
id269 | People with physical disabilities should not be allowed to use public transport. | hateful
id270 | The transport department is testing accessible buses across the city. | unhateful
id271 | People with learning differences should be denied admission to professional courses. | hateful
id272 | The college offers different learning formats to support its students. | unhateful
id273 | People who use mobility aids should be excluded from community gatherings. | hateful
id274 | The community centre has installed accessible entrances and seating. | unhateful
id275 | People with speech impairments should not be allowed to vote. | hateful
id276 | The election office provides accessible voting assistance. | unhateful
id277 | People with hearing loss should be denied access to emergency information. | hateful
id278 | The emergency department now provides visual alerts and written instructions. | unhateful
id279 | People with facial differences should be excluded from television programmes. | hateful
id280 | The television programme features interviews with people from varied backgrounds. | unhateful
id281 | Women should receive lower salaries than men for identical work. | hateful
id282 | The company is conducting an audit of pay differences between employees. | unhateful
id283 | Mothers should be permanently excluded from professional promotions. | hateful
id284 | The organisation has introduced parental leave for all eligible employees. | unhateful
id285 | Fathers should not be allowed to take responsibility for childcare. | hateful
id286 | The parenting workshop welcomes mothers, fathers, and guardians. | unhateful
id287 | Girls should be forbidden from enrolling in engineering programmes. | hateful
id288 | The engineering department is hosting an open day for school students. | unhateful
id289 | Boys from that community should not be allowed to study nursing. | hateful
id290 | The nursing college welcomes applicants of every gender. | unhateful
id291 | Non-binary people should be denied access to university facilities. | hateful
id292 | The university is updating its facilities to better serve all students. | unhateful
id293 | Intersex people should be excluded from ordinary community events. | hateful
id294 | The health seminar provides general information about variations in sex characteristics. | unhateful
id295 | Transgender employees should be dismissed regardless of their performance. | hateful
id296 | The company has published guidelines against workplace discrimination. | unhateful
id297 | Gay couples should be refused accommodation at every hotel. | hateful
id298 | The hotel has updated its booking policy to treat all guests equally. | unhateful
id299 | Lesbian students should be prohibited from joining the student council. | hateful
id300 | The student council is discussing ways to make campus activities more inclusive. | unhateful
id301 | Bisexual people should not be trusted with positions of responsibility. | hateful
id302 | The article explains common misconceptions about bisexuality. | unhateful
id303 | People who are questioning their identity should be excluded from school clubs. | hateful
id304 | The school counsellor provides confidential guidance to students. | unhateful
id305 | Men who choose nursing careers should be denied professional recognition. | hateful
id306 | The hospital has recruited nurses from a wide range of backgrounds. | unhateful
id307 | Women should be barred from participating in local elections. | hateful
id308 | The election commission has published information about voter registration. | unhateful
id309 | People of a particular gender should not be allowed to own property. | hateful
id310 | The legal aid centre provides information about property ownership rights. | unhateful
id311 | Older workers should be rejected regardless of their experience. | hateful
id312 | The company is reviewing retirement and succession policies. | unhateful
id313 | People above sixty should lose their right to participate in elections. | hateful
id314 | The government is studying changes to pension eligibility. | unhateful
id315 | Teenagers from that ethnic community should not be allowed into public libraries. | hateful
id316 | The public library has launched a reading programme for teenagers. | unhateful
id317 | Children from poor families should be excluded from school competitions. | hateful
id318 | The school has introduced fee assistance for students from low-income families. | unhateful
id319 | People from wealthy families should be denied basic legal protection. | hateful
id320 | The legal system should provide equal protection regardless of income. | unhateful
id321 | People from rural communities should not be permitted to work in urban offices. | hateful
id322 | The city is organising a career fair for rural graduates. | unhateful
id323 | Residents from villages should be refused admission to city colleges. | hateful
id324 | The college has started an orientation programme for students from rural areas. | unhateful
id325 | People with a particular regional accent should be rejected from customer service roles. | hateful
id326 | The training programme helps employees communicate clearly with customers. | unhateful
id327 | Speakers of minority languages should not receive public information in their language. | hateful
id328 | The department is translating important notices into several regional languages. | unhateful
id329 | People who follow a minority cultural tradition should be denied community hall access. | hateful
id330 | The community hall hosts events representing different cultural traditions. | unhateful
id331 | People from neighbouring countries should never be permitted to study here. | hateful
id332 | The university has students from several neighbouring countries. | unhateful
id333 | Foreign-born residents should be denied emergency medical assistance. | hateful
id334 | Emergency medical services are available to people who need urgent care. | unhateful
id335 | People with a different national origin should receive fewer workplace benefits. | hateful
id336 | The company is standardising employee benefits across its locations. | unhateful
id337 | People who speak with a foreign accent should be excluded from public meetings. | hateful
id338 | The public meeting will provide translation support for attendees. | unhateful
id339 | People from a particular country should not be allowed to compete in academic contests. | hateful
id340 | The academic competition is open to eligible students from all participating countries. | unhateful
id341 | Members of a minority religion should be denied the right to open places of worship. | hateful
id342 | The local authority is reviewing applications for new places of worship. | unhateful
id343 | People who wear religious clothing should be refused entry to public libraries. | hateful
id344 | The library policy permits visitors to wear their usual clothing. | unhateful
id345 | People who observe religious dietary rules should be excluded from school meals. | hateful
id346 | The school cafeteria is collecting feedback about dietary requirements. | unhateful
id347 | Religious minorities should not be permitted to celebrate their festivals publicly. | hateful
id348 | The city calendar includes public information about several religious festivals. | unhateful
id349 | People who do not follow the majority religion should not receive public scholarships. | hateful
id350 | The scholarship programme publishes eligibility criteria for every applicant. | unhateful
id351 | People who change their religion should lose their citizenship. | hateful
id352 | The constitution describes the legal process for changing personal records. | unhateful
id353 | Members of minority faiths should not be allowed to teach history. | hateful
id354 | The history department is revising its curriculum and reading list. | unhateful
id355 | People who practise traditional indigenous beliefs should be denied legal recognition. | hateful
id356 | The research team is documenting traditional indigenous knowledge. | unhateful
id357 | People who do not believe in any religion should be banned from public universities. | hateful
id358 | The philosophy course compares religious and non-religious viewpoints. | unhateful
id359 | Members of a particular caste should not be allowed to use community wells. | hateful
id360 | The village committee is repairing the shared water supply. | unhateful
id361 | Students from a scheduled caste should be prevented from using the school laboratory. | hateful
id362 | The school has introduced a common laboratory schedule for all classes. | unhateful
id363 | People from scheduled tribes should be denied access to government hostels. | hateful
id364 | The government has announced repairs to several student hostels. | unhateful
id365 | Caste background should determine which public facilities a person can use. | hateful
id366 | Public facilities should be accessible to residents without caste discrimination. | unhateful
id367 | People from a particular caste should not be allowed to share meals with others. | hateful
id368 | The school is organising a community lunch for all students. | unhateful
id369 | People from lower-status caste backgrounds should not be allowed to become teachers. | hateful
id370 | The education department is recruiting teachers for government schools. | unhateful
id371 | Caste identity should be used to reject job applicants automatically. | hateful
id372 | The recruitment form asks applicants about their education and work experience. | unhateful
id373 | People from a particular caste should be prevented from entering public parks. | hateful
id374 | The municipal corporation is renovating several public parks. | unhateful
id375 | Children from certain caste backgrounds should be seated separately in classrooms. | hateful
id376 | The school has arranged mixed seating groups for collaborative learning. | unhateful
id377 | A person's caste should decide whether they can access a hospital. | hateful
id378 | The hospital is introducing a digital appointment booking system. | unhateful
id379 | People from a particular caste should not be allowed to contest local elections. | hateful
id380 | The election office is preparing voter awareness materials. | unhateful
id381 | Members of an ethnic minority should be denied the right to establish businesses. | hateful
id382 | The business association is offering workshops to new entrepreneurs. | unhateful
id383 | People from a certain ethnic background should be refused public transport passes. | hateful
id384 | The transport authority has updated its student pass application process. | unhateful
id385 | Ethnic minority children should not be included in national school competitions. | hateful
id386 | The national school competition has opened registrations for eligible schools. | unhateful
id387 | People from a particular ethnic group should not be allowed to become doctors. | hateful
id388 | The medical college has announced its annual admissions schedule. | unhateful
id389 | People of mixed ethnic heritage should be denied citizenship. | hateful
id390 | The citizenship office has published updated application instructions. | unhateful
id391 | People from a minority ethnic background should not be allowed to represent the country in sports. | hateful
id392 | The sports federation is selecting athletes for the national team. | unhateful
id393 | Ethnic background should determine who can receive public housing. | hateful
id394 | The housing department is reviewing applications using published eligibility rules. | unhateful
id395 | People from a particular ethnic community should be excluded from cultural festivals. | hateful
id396 | The festival committee has invited artists from several cultural communities. | unhateful
id397 | Members of a minority group should not be permitted to use public swimming pools. | hateful
id398 | The swimming pool has published its opening hours and visitor rules. | unhateful
id399 | People from a particular ancestry should be barred from government employment. | hateful
id400 | The government has opened applications for several administrative positions. | unhateful
id401 | People with a specific disability should be denied access to voting booths. | hateful
id402 | Election officials are testing accessible voting equipment. | unhateful
id403 | People with visual disabilities should not be permitted to use digital services. | hateful
id404 | The website is being redesigned to support screen readers. | unhateful
id405 | People with hearing disabilities should be excluded from emergency response teams. | hateful
id406 | The emergency service is training staff in accessible communication. | unhateful
id407 | People with physical impairments should not be allowed to participate in school trips. | hateful
id408 | The school is planning accessible transport for its next educational trip. | unhateful
id409 | People with intellectual disabilities should be denied the right to make personal choices. | hateful
id410 | The support organisation provides guidance for independent decision-making. | unhateful
id411 | People with chronic illnesses should automatically be rejected from all employment. | hateful
id412 | The employer is reviewing workplace adjustments for staff with health needs. | unhateful
id413 | People with mobility impairments should be prohibited from attending concerts. | hateful
id414 | The concert venue has published information about accessible seating. | unhateful
id415 | People with speech differences should be excluded from public consultations. | hateful
id416 | The consultation accepts written feedback as well as spoken contributions. | unhateful
id417 | People with facial differences should be refused customer service. | hateful
id418 | The customer service team is introducing additional staff training. | unhateful
id419 | People with a history of mental illness should be denied educational opportunities. | hateful
id420 | The university provides academic support to students facing personal difficulties. | unhateful
id421 | People with neurological conditions should be barred from public libraries. | hateful
id422 | The library is adding quiet study spaces for visitors. | unhateful
id423 | People with learning disabilities should be refused access to vocational training. | hateful
id424 | The training centre offers courses with different levels of learning support. | unhateful
id425 | People with disabilities should not be allowed to participate in public consultations. | hateful
id426 | The council accepts feedback through online and in-person meetings. | unhateful
id427 | People who use assistive technology should be excluded from computer laboratories. | hateful
id428 | The computer laboratory is installing adaptive keyboards and other accessibility tools. | unhateful
id429 | People with disabilities should be paid less for equivalent work. | hateful
id430 | The organisation is reviewing its equal-pay policy. | unhateful
id431 | Women should be prevented from applying for senior research positions. | hateful
id432 | The research institute has announced vacancies across several departments. | unhateful
id433 | Women should not be allowed to inherit family property. | hateful
id434 | The legal clinic is hosting a session about inheritance law. | unhateful
id435 | Girls should be prohibited from participating in mathematics competitions. | hateful
id436 | The mathematics club is preparing students for an upcoming competition. | unhateful
id437 | Female students should not be permitted to study computer science. | hateful
id438 | The computer science department has published its course schedule. | unhateful
id439 | Women should be denied access to professional sports facilities. | hateful
id440 | The sports centre has extended its opening hours for all registered members. | unhateful
id441 | Men should be excluded from early-childhood education because of their gender. | hateful
id442 | The childcare centre is recruiting qualified educators. | unhateful
id443 | Boys should be denied access to art and music classes. | hateful
id444 | The school has added music and visual arts to its extracurricular programme. | unhateful
id445 | People who identify outside the gender binary should not be allowed to study abroad. | hateful
id446 | The international office is hosting a session about student exchange applications. | unhateful
id447 | Transgender people should be refused access to government identity services. | hateful
id448 | The government portal explains how residents can update their identity documents. | unhateful
id449 | Gay people should be prohibited from applying for teaching licences. | hateful
id450 | The licensing board has published updated teacher registration requirements. | unhateful
id451 | Lesbian people should not be allowed to rent office space. | hateful
id452 | The business centre has announced new office rental options. | unhateful
id453 | Bisexual people should be denied membership in professional organisations. | hateful
id454 | The professional association has opened its annual membership registration. | unhateful
id455 | People of a particular sexual orientation should be refused public healthcare. | hateful
id456 | The health department has published a directory of public clinics. | unhateful
id457 | People who are questioning their gender should be denied counselling services. | hateful
id458 | The counselling centre offers appointments to students seeking guidance. | unhateful
id459 | People who are intersex should be excluded from university research programmes. | hateful
id460 | The university has announced a new research ethics seminar. | unhateful
id461 | People from a particular age group should be denied the right to participate in community decisions. | hateful
id462 | The residents' association is collecting suggestions from people of all ages. | unhateful
id463 | Older adults should be refused access to digital literacy classes. | hateful
id464 | The library is offering beginner computer classes for older adults. | unhateful
id465 | Young adults should be excluded from local employment schemes because of their age. | hateful
id466 | The employment centre is organising a career guidance session for graduates. | unhateful
id467 | Children from families with low incomes should not receive school supplies. | hateful
id468 | The charity is distributing notebooks and stationery to students. | unhateful
id469 | People experiencing homelessness should be prohibited from entering public parks. | hateful
id470 | The city is reviewing its services for people experiencing homelessness. | unhateful
id471 | People from a particular social class should not be allowed to use public hospitals. | hateful
id472 | The hospital has introduced a new system for managing patient appointments. | unhateful
id473 | People who are unemployed should be denied access to public libraries. | hateful
id474 | The library offers free internet access to visitors. | unhateful
id475 | People from rural backgrounds should not be permitted to apply for civil service jobs. | hateful
id476 | The civil service examination board has published its application calendar. | unhateful
id477 | People from urban communities should be excluded from agricultural training programmes. | hateful
id478 | The agricultural institute is offering training to interested applicants. | unhateful
id479 | People who speak a regional dialect should not be allowed to participate in court proceedings. | hateful
id480 | Courts can arrange interpretation support for people who need it. | unhateful
id481 | People who belong to a linguistic minority should not be allowed to publish books. | hateful
id482 | The publisher has announced a competition for new authors. | unhateful
id483 | Migrant workers from a particular country should be denied payment for their labour. | hateful
id484 | The labour department is reviewing complaints about unpaid wages. | unhateful
id485 | Foreign students should be excluded from university research facilities solely because of their nationality. | hateful
id486 | The university has published safety instructions for all laboratory users. | unhateful
id487 | People from a neighbouring country should not be allowed to receive legal representation. | hateful
id488 | The legal aid office has expanded its public consultation hours. | unhateful
id489 | People who speak a minority language should be denied the right to register businesses. | hateful
id490 | The business portal now supports applications in multiple languages. | unhateful
id491 | People from a particular country should be refused entry to public museums. | hateful
id492 | The museum is introducing free admission on selected public holidays. | unhateful
id493 | People of a minority faith should not be allowed to work in public hospitals. | hateful
id494 | The hospital is recruiting staff for its new outpatient department. | unhateful
id495 | Members of a religious minority should be prevented from participating in school elections. | hateful
id496 | The school has announced the dates for its student council elections. | unhateful
id497 | People who wear traditional cultural clothing should be denied entry to government offices. | hateful
id498 | The government office has updated its visitor information page. | unhateful
id499 | People from a particular caste should be refused access to public drinking water. | hateful
id500 | The village administration is testing the quality of its drinking water. | unhateful
id501 | People from a minority ethnic group should be excluded from public sports teams. | hateful
id502 | The sports department has announced open trials for its district teams. | unhateful
id503 | People of a particular race should not be allowed to receive academic awards. | hateful
id504 | The university has announced the criteria for its annual academic awards. | unhateful
id505 | People from a particular ancestry should be prohibited from serving on community committees. | hateful
id506 | The community committee is inviting residents to volunteer for local projects. | unhateful
id507 | People of a particular religion should not be allowed to use public transportation during festivals. | hateful
id508 | The transport authority is adding extra buses during the festival period. | unhateful
id509 | People with a particular disability should not be allowed to attend public examinations. | hateful
id510 | The examination board has published arrangements for candidates requiring accessibility support. | unhateful
id511 | Women from a particular ethnic background should not be allowed to lead research teams. | hateful
id512 | The research conference has invited speakers from several academic disciplines. | unhateful
id513 | Men from a particular nationality should be denied access to childcare services. | hateful
id514 | The childcare department is reviewing its registration procedures. | unhateful
id515 | People of a particular gender should not be permitted to serve on a jury. | hateful
id516 | The court has published general information about jury selection. | unhateful
id517 | People from a particular community should be denied the opportunity to participate in public art projects. | hateful
id518 | The city is inviting residents to submit ideas for a public art installation. | unhateful
id519 | People who follow minority cultural practices should be refused entry to community centres. | hateful
id520 | The community centre has published a calendar of cultural activities. | unhateful
id521 | People from a particular ethnic group should not be allowed to receive disaster relief. | hateful
id522 | The relief agency is distributing supplies according to assessed need. | unhateful
id523 | People of a particular religion should be denied access to government education programmes. | hateful
id524 | The education department is expanding its adult literacy programme. | unhateful
id525 | People from a particular caste should be excluded from professional associations. | hateful
id526 | The professional association is organising a seminar on workplace ethics. | unhateful
id527 | People of a particular nationality should be denied equal protection under employment law. | hateful
id528 | The labour office has published a guide to employee rights. | unhateful
id529 | People from a particular ethnic community should be prevented from participating in local cultural decisions. | hateful
id530 | The local council is asking residents to comment on its cultural development plan. | unhateful"""

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

with open("scratch/eval_batch2_results.json", "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, indent=2)

print("\n" + "="*50)
print("BATCH 2 EVALUATION COMPLETED (320 SAMPLES):")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Macro F1: {f1_m:.4f}")
print(f"Macro Precision: {prec_m:.4f} | Macro Recall: {rec_m:.4f}")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Total Mismatches: {len(mismatches)}")
print("="*50)


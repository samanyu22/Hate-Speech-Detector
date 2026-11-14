import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from src.models.transformer_classifier import MultilingualTransformerEngine

raw_data = """nid1 | People from that ethnic group should be prohibited from renting public housing. | hateful
nid2 | The housing department has published its revised rental application guidelines. | unhateful
nid3 | Members of that religious community should be excluded from local business associations. | hateful
nid4 | The business association is organising a networking event for local entrepreneurs. | unhateful
nid5 | People of that caste should not be permitted to enrol in government training institutes. | hateful
nid6 | The training institute has introduced three new technical courses. | unhateful
nid7 | People with disabilities should be refused entry to public swimming pools. | hateful
nid8 | The recreation department is renovating the municipal swimming pool. | unhateful
nid9 | Women from that community should not be allowed to manage public financial institutions. | hateful
nid10 | The finance department has released its quarterly expenditure report. | unhateful
nid11 | People of that nationality should be barred from joining public research projects. | hateful
nid12 | The research council has announced its annual project selection schedule. | unhateful
nid13 | Members of that minority religion should not be allowed to operate food businesses. | hateful
nid14 | The food safety office has updated its restaurant inspection calendar. | unhateful
nid15 | People from that ethnic background should be excluded from government housing committees. | hateful
nid16 | The housing committee is reviewing applications for its next meeting. | unhateful
nid17 | People with hearing impairments should be denied access to public museums. | hateful
nid18 | The museum has added audio guides and updated its visitor information. | unhateful
nid19 | People from that caste should not be allowed to become registered architects. | hateful
nid20 | The architecture council has opened its annual membership renewal process. | unhateful
nid21 | People who follow that faith should be prevented from applying for public sector jobs. | hateful
nid22 | The recruitment office has published a list of available public sector vacancies. | unhateful
nid23 | People from that country should be excluded from university sports teams. | hateful
nid24 | The university sports department has announced its inter-college tournament dates. | unhateful
nid25 | People with visual impairments should not be permitted to use public transport. | hateful
nid26 | The transport authority is testing improved audio announcements at bus stations. | unhateful
nid27 | Members of that ethnic community should be denied access to public parks. | hateful
nid28 | The parks department has announced the opening of a new botanical garden. | unhateful
nid29 | People of that gender should not be allowed to lead government departments. | hateful
nid30 | The government has published its latest departmental performance report. | unhateful
nid31 | People from that religious group should be prohibited from owning agricultural businesses. | hateful
nid32 | The agriculture office is hosting a seminar on sustainable farming methods. | unhateful
nid33 | People of that caste should not be permitted to join public medical colleges. | hateful
nid34 | The medical college has released its updated admission calendar. | unhateful
nid35 | People with mobility disabilities should be excluded from public voting facilities. | hateful
nid36 | The election office is reviewing accessibility arrangements at polling stations. | unhateful
nid37 | People from that nationality should not be allowed to receive government research funding. | hateful
nid38 | The research department has announced a new funding application window. | unhateful
nid39 | Members of that cultural group should be barred from participating in city festivals. | hateful
nid40 | The cultural department is preparing its annual festival programme. | unhateful
nid41 | People from that ethnic background should not be allowed to work in public schools. | hateful
nid42 | The education department has released its teacher recruitment schedule. | unhateful
nid43 | People of that religion should be excluded from public legal advisory boards. | hateful
nid44 | The legal advisory board is reviewing its public consultation process. | unhateful
nid45 | People with intellectual disabilities should be denied access to public libraries. | hateful
nid46 | The library has introduced a new collection of accessible learning materials. | unhateful
nid47 | People from that caste should not be allowed to join village development committees. | hateful
nid48 | The village committee is collecting suggestions for its next development plan. | unhateful
nid49 | People from that country should be prohibited from applying for public health grants. | hateful
nid50 | The health department has published its latest community grant guidelines. | unhateful
nid51 | People of that sexual orientation should not be allowed to work in government offices. | hateful
nid52 | The administrative office has introduced a new employee training schedule. | unhateful
nid53 | People with autism should be excluded from public technology competitions. | hateful
nid54 | The technology competition has opened registration for student participants. | unhateful
nid55 | People from that minority language group should not be permitted to join broadcasting teams. | hateful
nid56 | The broadcasting service is accepting proposals for regional programmes. | unhateful
nid57 | People from that ethnic group should be refused membership in public cooperatives. | hateful
nid58 | The cooperative society has published its updated membership application form. | unhateful
nid59 | Older people from that community should not be allowed to attend public training sessions. | hateful
nid60 | The community centre has scheduled computer training sessions for next month. | unhateful
nid61 | People of that religion should be barred from receiving government housing assistance. | hateful
nid62 | The housing office has opened applications for its new assistance scheme. | unhateful
nid63 | People with speech disabilities should not be permitted to participate in public interviews. | hateful
nid64 | The recruitment team has published guidance for its upcoming interview process. | unhateful
nid65 | People from that nationality should be excluded from professional engineering bodies. | hateful
nid66 | The engineering association is hosting a conference on renewable energy. | unhateful
nid67 | Members of that caste should not be allowed to use community health centres. | hateful
nid68 | The community health centre has extended its consultation hours. | unhateful
nid69 | Women from that ethnic background should be prohibited from applying for leadership positions. | hateful
nid70 | The leadership programme is accepting applications from interested professionals. | unhateful
nid71 | People from that region should not be allowed to participate in national science fairs. | hateful
nid72 | The science fair committee has announced its project submission deadline. | unhateful
nid73 | People with learning disabilities should be denied access to public employment services. | hateful
nid74 | The employment office has launched a new online job search portal. | unhateful
nid75 | Members of that faith should not be allowed to serve on school governing boards. | hateful
nid76 | The school board is reviewing proposals for its next academic year. | unhateful
nid77 | People from that ethnic community should be excluded from public financial aid programmes. | hateful
nid78 | The financial aid office has updated its student support information. | unhateful
nid79 | People of that gender identity should not be permitted to use public sports grounds. | hateful
nid80 | The sports authority has announced new booking slots for its grounds. | unhateful
nid81 | People from that caste should not be allowed to become licensed medical practitioners. | hateful
nid82 | The medical council has published its revised licensing examination dates. | unhateful
nid83 | People with neurological disabilities should be prohibited from joining public research teams. | hateful
nid84 | The research institute has announced vacancies for laboratory assistants. | unhateful
nid85 | People from that nationality should be denied access to public legal assistance. | hateful
nid86 | The legal services office has opened a new appointment booking facility. | unhateful
nid87 | Members of that religious group should not be allowed to participate in local elections. | hateful
nid88 | The election commission has released its voter registration timetable. | unhateful
nid89 | People from that cultural background should be excluded from public heritage projects. | hateful
nid90 | The heritage department is documenting traditional buildings in the district. | unhateful
nid91 | People with physical disabilities should not be allowed to apply for public scholarships. | hateful
nid92 | The scholarship office has published its eligibility and submission guidelines. | unhateful
nid93 | People from that ethnic group should be prohibited from joining public emergency response teams. | hateful
nid94 | The emergency services department is recruiting community volunteers. | unhateful
nid95 | People of that religion should not be permitted to operate shops in public markets. | hateful
nid96 | The market authority has announced revised stall allocation dates. | unhateful
nid97 | People from that caste should be excluded from public water management committees. | hateful
nid98 | The water board is inviting residents to discuss conservation measures. | unhateful
nid99 | People from that country should not be allowed to participate in public arts funding programmes. | hateful
nid100 | The arts council has announced its next round of project funding. | unhateful
nid101 | People from that immigrant community should be barred from applying for public teaching positions. | hateful
nid102 | The education board has published the requirements for its teaching vacancies. | unhateful
nid103 | Members of that religion should be denied access to public disaster relief services. | hateful
nid104 | The disaster management office has released its monsoon preparedness advisory. | unhateful
nid105 | People with visual disabilities should not be allowed to register for public examinations. | hateful
nid106 | The examination board has opened online registration for its upcoming tests. | unhateful
nid107 | People of that nationality should be excluded from government technology programmes. | hateful
nid108 | The technology department has launched a digital skills development initiative. | unhateful
nid109 | People from that caste should not be permitted to join public environmental organisations. | hateful
nid110 | The environmental organisation is planning a tree plantation campaign. | unhateful
nid111 | People of that gender should be prohibited from receiving public leadership awards. | hateful
nid112 | The awards committee has published its nomination procedure. | unhateful
nid113 | People from that ethnic group should be refused access to public childcare services. | hateful
nid114 | The childcare office has announced registration dates for its new centres. | unhateful
nid115 | People with hearing disabilities should not be allowed to attend public university lectures. | hateful
nid116 | The university is updating lecture recordings with captions. | unhateful
nid117 | People from that religious community should be excluded from public cultural grants. | hateful
nid118 | The cultural office has published its grant application instructions. | unhateful
nid119 | People from that country should not be permitted to join national volunteer programmes. | hateful
nid120 | The volunteer organisation has opened applications for its annual programme. | unhateful
nid121 | People of that caste should be barred from registering property in the district. | hateful
nid122 | The property office has introduced an online document verification service. | unhateful
nid123 | People with mobility disabilities should not be allowed to attend public job fairs. | hateful
nid124 | The career centre is organising a job fair with employers from several sectors. | unhateful
nid125 | People from that ethnic background should be excluded from public housing consultations. | hateful
nid126 | The city council is inviting residents to comment on its housing proposal. | unhateful
nid127 | People of that religion should not be allowed to receive public agricultural loans. | hateful
nid128 | The agriculture bank has updated its loan application information. | unhateful
nid129 | People from that nationality should be prohibited from working in public laboratories. | hateful
nid130 | The laboratory has announced its annual equipment maintenance schedule. | unhateful
nid131 | People with intellectual disabilities should not be permitted to join community sports clubs. | hateful
nid132 | The community sports club has opened registration for its winter activities. | unhateful
nid133 | People from that caste should be denied access to public business incubators. | hateful
nid134 | The business incubator is accepting proposals from early-stage companies. | unhateful
nid135 | People of that sexual orientation should be excluded from public student housing. | hateful
nid136 | The student housing office has published its room allocation dates. | unhateful
nid137 | People from that ethnic community should not be allowed to participate in public policy discussions. | hateful
nid138 | The policy department is holding an open discussion about transport planning. | unhateful
nid139 | People who follow that faith should be barred from public professional examinations. | hateful
nid140 | The professional examination board has announced its revised test schedule. | unhateful
nid141 | People with autism should not be allowed to apply for public arts residencies. | hateful
nid142 | The arts centre has opened applications for its upcoming residency programme. | unhateful
nid143 | People from that nationality should be denied membership in public library committees. | hateful
nid144 | The library committee is reviewing proposals for its annual reading programme. | unhateful
nid145 | People of that gender identity should be excluded from public vocational courses. | hateful
nid146 | The vocational institute has added evening classes to its course schedule. | unhateful
nid147 | People from that region should not be permitted to receive government innovation grants. | hateful
nid148 | The innovation office has published details of its latest funding round. | unhateful
nid149 | People with speech disabilities should be prohibited from joining public debate competitions. | hateful
nid150 | The debate society has announced the topics for its next competition. | unhateful
nid151 | People from that ethnic group should not be allowed to become public accountants. | hateful
nid152 | The accounting institute has released its professional examination calendar. | unhateful
nid153 | Members of that caste should be excluded from public community kitchens. | hateful
nid154 | The community kitchen is updating its weekly meal distribution timetable. | unhateful
nid155 | People from that religious background should not be allowed to receive public pension benefits. | hateful
nid156 | The pension office has introduced a new online application tracking service. | unhateful
nid157 | People with physical disabilities should be denied access to public computer laboratories. | hateful
nid158 | The computer laboratory has upgraded its equipment for the new semester. | unhateful
nid159 | People from that country should not be permitted to join international academic projects. | hateful
nid160 | The academic office is coordinating a research exchange with partner universities. | unhateful
nid161 | People of that caste should be prohibited from joining public construction projects. | hateful
nid162 | The public works department has announced its upcoming road repair schedule. | unhateful
nid163 | People from that minority religion should not be allowed to use public community halls. | hateful
nid164 | The community hall has introduced a new online booking calendar. | unhateful
nid165 | People with hearing impairments should be excluded from public language courses. | hateful
nid166 | The language centre has announced its new beginner-level classes. | unhateful
nid167 | People from that nationality should not be permitted to join public conservation projects. | hateful
nid168 | The conservation team is inviting volunteers to help monitor local wildlife. | unhateful
nid169 | People of that ethnic background should be denied access to public legal training programmes. | hateful
nid170 | The legal training institute has published its upcoming workshop schedule. | unhateful
nid171 | People of that gender should not be allowed to become university department heads. | hateful
nid172 | The university has announced the process for appointing department heads. | unhateful
nid173 | People from that caste should be barred from participating in public agricultural fairs. | hateful
nid174 | The agricultural fair committee has released its exhibitor registration form. | unhateful
nid175 | People with learning disabilities should not be allowed to use public digital services. | hateful
nid176 | The digital services office is testing a simpler online application interface. | unhateful
nid177 | People from that ethnic group should be excluded from public tourism committees. | hateful
nid178 | The tourism department is collecting feedback about local visitor facilities. | unhateful
nid179 | People from that religion should not be permitted to join public education boards. | hateful
nid180 | The education board has scheduled a meeting to review school infrastructure. | unhateful
nid181 | People of that nationality should be denied access to public entrepreneurship support. | hateful
nid182 | The entrepreneurship office is hosting a workshop on business planning. | unhateful
nid183 | People with visual impairments should be prohibited from participating in public photography exhibitions. | hateful
nid184 | The photography exhibition is accepting entries from local artists. | unhateful
nid185 | People from that cultural community should not be allowed to join public museum boards. | hateful
nid186 | The museum board is reviewing its visitor engagement strategy. | unhateful
nid187 | People of that sexual orientation should be excluded from public mentoring programmes. | hateful
nid188 | The mentoring organisation has opened applications for its next cohort. | unhateful
nid189 | People from that caste should not be permitted to receive public environmental awards. | hateful
nid190 | The environmental council has announced nominations for its annual awards. | unhateful
nid191 | People from that immigrant background should be denied access to public career counselling. | hateful
nid192 | The career counselling centre has introduced online appointment booking. | unhateful
nid193 | People with neurological disabilities should not be allowed to participate in public science exhibitions. | hateful
nid194 | The science centre is preparing interactive exhibits for its annual event. | unhateful
nid195 | People from that ethnic group should be prohibited from joining public housing cooperatives. | hateful
nid196 | The housing cooperative is accepting applications from new residents. | unhateful
nid197 | People of that religion should not be allowed to participate in public heritage preservation programmes. | hateful
nid198 | The heritage office is recruiting volunteers to catalogue historic documents. | unhateful
nid199 | People from that nationality should be excluded from public scholarship selection panels. | hateful
nid200 | The scholarship committee has announced the date of its next selection meeting. | unhateful
nid201 | People from that indigenous community should be denied permission to establish cultural centres. | hateful
nid202 | The cultural department has opened applications for community centre funding. | unhateful
nid203 | Members of that religious group should not be allowed to register charitable organisations. | hateful
nid204 | The registration office has published new guidelines for charitable organisations. | unhateful
nid205 | People from that ethnic background should be prevented from applying for government apprenticeships. | hateful
nid206 | The apprenticeship programme has announced its next intake of trainees. | unhateful
nid207 | People of that caste should not be permitted to access public financial counselling. | hateful
nid208 | The financial counselling office is offering free sessions to residents. | unhateful
nid209 | People with disabilities should be excluded from public university research scholarships. | hateful
nid210 | The university has announced scholarships for postgraduate research students. | unhateful
nid211 | People from that nationality should not be allowed to participate in national literary competitions. | hateful
nid212 | The literary association has announced the theme of its annual writing competition. | unhateful
nid213 | People belonging to that faith should be prohibited from using public recreational facilities. | hateful
nid214 | The recreation department has published its updated facility booking rules. | unhateful
nid215 | People from that ethnic group should be refused access to public disaster shelters. | hateful
nid216 | The district administration has identified additional locations for emergency shelters. | unhateful
nid217 | People of that gender should not be allowed to receive professional teaching licences. | hateful
nid218 | The teaching council has updated its teacher certification requirements. | unhateful
nid219 | People from that caste should be excluded from public cooperative banking services. | hateful
nid220 | The cooperative bank has introduced a new mobile banking facility. | unhateful
nid221 | People with visual impairments should not be permitted to access public information kiosks. | hateful
nid222 | The municipal office has installed new information kiosks across the city. | unhateful
nid223 | People from that religious community should be barred from joining public cultural committees. | hateful
nid224 | The cultural committee is inviting residents to suggest events for the coming year. | unhateful
nid225 | People from that country should not be allowed to participate in university exchange programmes. | hateful
nid226 | The university exchange office has published its international programme calendar. | unhateful
nid227 | People with hearing impairments should be prohibited from applying for public broadcasting jobs. | hateful
nid228 | The broadcasting organisation has announced vacancies in its production department. | unhateful
nid229 | People of that caste should not be allowed to operate licensed pharmacies. | hateful
nid230 | The pharmacy council has published its updated registration requirements. | unhateful
nid231 | Members of that ethnic group should be excluded from public urban development projects. | hateful
nid232 | The urban development authority has invited proposals for a new housing project. | unhateful
nid233 | People from that minority religion should not be allowed to participate in public university elections. | hateful
nid234 | The university election committee has released its candidate nomination schedule. | unhateful
nid235 | People with mobility disabilities should be refused access to public community gardens. | hateful
nid236 | The gardening association is distributing free seeds to neighbourhood residents. | unhateful
nid237 | People from that nationality should be barred from applying for government research positions. | hateful
nid238 | The research department has published its latest recruitment notification. | unhateful
nid239 | People of that gender identity should not be permitted to join public professional networks. | hateful
nid240 | The professional network is hosting an online career development seminar. | unhateful
nid241 | People from that ethnic community should be denied access to government-funded art studios. | hateful
nid242 | The arts department has opened registrations for its studio workshops. | unhateful
nid243 | People from that caste should not be allowed to participate in public irrigation projects. | hateful
nid244 | The irrigation department has scheduled a consultation with local farmers. | unhateful
nid245 | People who follow that religion should be excluded from public disaster response training. | hateful
nid246 | The disaster response authority is organising a first-aid training programme. | unhateful
nid247 | People with intellectual disabilities should not be permitted to join public music programmes. | hateful
nid248 | The music academy has announced auditions for its upcoming student orchestra. | unhateful
nid249 | People from that nationality should be prohibited from receiving public innovation awards. | hateful
nid250 | The innovation council has announced its annual award nomination process. | unhateful
nid251 | People from that ethnic background should not be allowed to access government legal databases. | hateful
nid252 | The legal department has upgraded its online document search system. | unhateful
nid253 | Members of that caste should be denied permission to establish educational trusts. | hateful
nid254 | The education department has published procedures for registering educational trusts. | unhateful
nid255 | People of that religion should not be allowed to participate in public environmental campaigns. | hateful
nid256 | The environmental agency is launching a campaign to reduce plastic waste. | unhateful
nid257 | People with speech disabilities should be excluded from public customer service positions. | hateful
nid258 | The customer service department has introduced a new staff training module. | unhateful
nid259 | People from that country should not be permitted to receive government startup assistance. | hateful
nid260 | The startup support office is accepting applications for its business mentoring scheme. | unhateful
nid261 | People of that gender should be prohibited from applying for public engineering scholarships. | hateful
nid262 | The engineering college has announced its scholarship application dates. | unhateful
nid263 | People from that cultural community should not be allowed to participate in public archaeological surveys. | hateful
nid264 | The archaeology department is recruiting volunteers for a historical survey. | unhateful
nid265 | People with neurological disabilities should be barred from accessing public vocational workshops. | hateful
nid266 | The vocational centre has published its workshop timetable for December. | unhateful
nid267 | People belonging to that religious group should be excluded from public housing lotteries. | hateful
nid268 | The housing authority has announced the schedule for its next allocation lottery. | unhateful
nid269 | People from that caste should not be permitted to join public agricultural research boards. | hateful
nid270 | The agricultural research board has released its annual findings. | unhateful
nid271 | People from that nationality should be refused membership in public scientific societies. | hateful
nid272 | The scientific society is accepting membership applications from researchers. | unhateful
nid273 | People with autism should not be allowed to access public employment training centres. | hateful
nid274 | The employment centre has introduced a new skills assessment programme. | unhateful
nid275 | People from that ethnic group should be excluded from public heritage tourism projects. | hateful
nid276 | The tourism department is developing a new historical walking route. | unhateful
nid277 | People of that religion should not be permitted to apply for public childcare subsidies. | hateful
nid278 | The childcare department has published details of its subsidy programme. | unhateful
nid279 | People from that caste should be prohibited from participating in public construction tenders. | hateful
nid280 | The construction authority has released its latest tender notices. | unhateful
nid281 | People with physical disabilities should be denied access to public legal education programmes. | hateful
nid282 | The legal education centre is offering introductory courses on citizens' rights. | unhateful
nid283 | People from that nationality should not be allowed to join government climate research teams. | hateful
nid284 | The climate research institute has published its latest project summary. | unhateful
nid285 | Members of that religious community should be excluded from public transport advisory groups. | hateful
nid286 | The transport authority is seeking feedback on proposed bus route changes. | unhateful
nid287 | People from that ethnic background should not be permitted to use public financial assistance portals. | hateful
nid288 | The finance department has redesigned its online assistance portal. | unhateful
nid289 | People of that gender identity should be prohibited from participating in public leadership workshops. | hateful
nid290 | The leadership institute has opened registration for its next workshop. | unhateful
nid291 | People with hearing disabilities should not be allowed to apply for public library positions. | hateful
nid292 | The library service has announced recruitment for several vacant positions. | unhateful
nid293 | People from that caste should be excluded from government-sponsored entrepreneurship contests. | hateful
nid294 | The entrepreneurship department has announced a competition for new business ideas. | unhateful
nid295 | People of that religion should not be permitted to access public mental wellness programmes. | hateful
nid296 | The wellness centre has published its schedule of free community sessions. | unhateful
nid297 | People from that ethnic group should be barred from joining public sports coaching programmes. | hateful
nid298 | The sports academy has opened applications for its coaching certification course. | unhateful
nid299 | People with learning disabilities should not be allowed to participate in public language examinations. | hateful
nid300 | The language examination board has published its updated test dates. | unhateful
nid301 | People from that country should be excluded from government-funded clean energy projects. | hateful
nid302 | The energy department has announced funding for renewable energy research. | unhateful
nid303 | Members of that caste should not be allowed to register as public building contractors. | hateful
nid304 | The building authority has updated its contractor registration portal. | unhateful
nid305 | People from that religious community should be denied access to public legal clinics. | hateful
nid306 | The legal clinic has announced additional consultation hours for residents. | unhateful
nid307 | People of that nationality should not be permitted to apply for public museum fellowships. | hateful
nid308 | The museum has opened applications for its historical research fellowship. | unhateful
nid309 | People with visual impairments should be excluded from public map-reading workshops. | hateful
nid310 | The geography club is organising a workshop on digital mapping tools. | unhateful
nid311 | People from that ethnic group should not be allowed to join public education research panels. | hateful
nid312 | The education research panel is inviting applications from interested academics. | unhateful
nid313 | People from that caste should be barred from using government agricultural marketplaces. | hateful
nid314 | The agricultural marketplace has launched an online produce listing service. | unhateful
nid315 | People who follow that faith should not be allowed to access public housing advice centres. | hateful
nid316 | The housing advice centre is holding a free information session for applicants. | unhateful
nid317 | People of that gender should be excluded from public technology leadership programmes. | hateful
nid318 | The technology institute has announced its annual leadership development course. | unhateful
nid319 | People from that nationality should be denied access to public maritime training. | hateful
nid320 | The maritime institute has published its new training course catalogue. | unhateful
nid321 | People with mobility disabilities should not be allowed to participate in public cycling events. | hateful
nid322 | The cycling association has announced the route for its annual city ride. | unhateful
nid323 | People from that ethnic background should be prohibited from receiving government cultural fellowships. | hateful
nid324 | The cultural foundation has published its fellowship application criteria. | unhateful
nid325 | People belonging to that caste should not be permitted to join public disaster planning committees. | hateful
nid326 | The disaster planning committee is reviewing the district emergency response plan. | unhateful
nid327 | People from that religious group should be excluded from public university accommodation programmes. | hateful
nid328 | The university accommodation office has announced its hostel application dates. | unhateful
nid329 | People with intellectual disabilities should not be allowed to access public art education. | hateful
nid330 | The art school has introduced weekend classes for beginners. | unhateful
nid331 | People from that nationality should be prohibited from participating in public scientific exhibitions. | hateful
nid332 | The science exhibition committee has invited schools to register their projects. | unhateful
nid333 | People of that sexual orientation should be denied access to public community leadership programmes. | hateful
nid334 | The community leadership programme has published its upcoming session dates. | unhateful
nid335 | People from that ethnic group should not be allowed to join public library reading clubs. | hateful
nid336 | The library has started a monthly reading club for local residents. | unhateful
nid337 | People with neurological disabilities should be excluded from government-funded coding courses. | hateful
nid338 | The coding academy has opened registration for its beginner programming course. | unhateful
nid339 | People from that caste should not be permitted to receive public rural development grants. | hateful
nid340 | The rural development office has announced its latest grant application period. | unhateful
nid341 | People of that religion should be barred from joining public university alumni associations. | hateful
nid342 | The alumni association is organising its annual networking event. | unhateful
nid343 | People from that country should not be allowed to access public employment training schemes. | hateful
nid344 | The employment department has introduced a new workplace skills programme. | unhateful
nid345 | People with hearing impairments should be denied access to public driving theory classes. | hateful
nid346 | The driving school has published its timetable for theory classes. | unhateful
nid347 | People from that ethnic background should be excluded from public food distribution committees. | hateful
nid348 | The food distribution office has published its monthly supply schedule. | unhateful
nid349 | People from that caste should not be permitted to participate in public water conservation competitions. | hateful
nid350 | The water authority has announced a competition for school conservation projects. | unhateful
nid351 | People of that gender identity should be prohibited from applying for public research fellowships. | hateful
nid352 | The research institute has opened applications for its summer fellowship. | unhateful
nid353 | People belonging to that faith should not be allowed to join public neighbourhood safety groups. | hateful
nid354 | The neighbourhood safety group is organising a meeting about street lighting. | unhateful
nid355 | People with visual impairments should be excluded from public digital literacy courses. | hateful
nid356 | The digital literacy centre has announced free computer classes for residents. | unhateful
nid357 | People from that nationality should not be allowed to become members of public accounting bodies. | hateful
nid358 | The accounting body has published its annual membership renewal notice. | unhateful
nid359 | People from that ethnic community should be denied access to public cultural archives. | hateful
nid360 | The cultural archive has digitised a collection of historical photographs. | unhateful
nid361 | People of that caste should be prohibited from participating in public forest management programmes. | hateful
nid362 | The forest department is inviting residents to attend a conservation meeting. | unhateful
nid363 | People from that religion should not be allowed to receive public educational loans. | hateful
nid364 | The education finance office has updated its student loan application portal. | unhateful
nid365 | People with speech disabilities should be excluded from public radio presentation courses. | hateful
nid366 | The radio academy has announced a new course in audio production. | unhateful
nid367 | People from that country should not be permitted to join public medical research networks. | hateful
nid368 | The medical research network has announced its annual conference programme. | unhateful
nid369 | People from that ethnic group should be barred from participating in public housing renovation schemes. | hateful
nid370 | The housing department has announced a renovation programme for older buildings. | unhateful
nid371 | People of that gender should not be allowed to join public science advisory councils. | hateful
nid372 | The science council is seeking nominations for its next advisory committee. | unhateful
nid373 | People from that caste should be denied access to public agricultural training farms. | hateful
nid374 | The agricultural institute has scheduled practical demonstrations at its training farm. | unhateful
nid375 | People with autism should be prohibited from participating in public robotics clubs. | hateful
nid376 | The robotics club is accepting registrations for its next student workshop. | unhateful
nid377 | People belonging to that religious community should not be permitted to join public legal associations. | hateful
nid378 | The legal association is hosting a seminar on recent court decisions. | unhateful
nid379 | People from that nationality should be excluded from public entrepreneurship competitions. | hateful
nid380 | The entrepreneurship competition has published its submission guidelines. | unhateful
nid381 | People from that ethnic background should not be allowed to access government-funded music education. | hateful
nid382 | The music department has announced free lessons at its community centres. | unhateful
nid383 | People with physical disabilities should be refused entry to public science laboratories. | hateful
nid384 | The science laboratory has published its visitor safety instructions. | unhateful
nid385 | People of that religion should not be permitted to participate in public city planning surveys. | hateful
nid386 | The planning department has opened an online survey about future road improvements. | unhateful
nid387 | People from that caste should be prohibited from joining public health advisory committees. | hateful
nid388 | The health advisory committee has published minutes from its latest meeting. | unhateful
nid389 | People from that nationality should be denied access to public professional development courses. | hateful
nid390 | The professional development centre has announced its next training calendar. | unhateful
nid391 | People with learning disabilities should not be allowed to join public photography classes. | hateful
nid392 | The photography centre has opened registration for its weekend classes. | unhateful
nid393 | People from that ethnic group should be excluded from public renewable energy projects. | hateful
nid394 | The energy institute has announced a community solar research project. | unhateful
nid395 | People of that gender identity should not be permitted to participate in public student mentoring schemes. | hateful
nid396 | The student mentoring office has opened registration for its new programme. | unhateful
nid397 | People from that religious background should be barred from joining public cultural exchange programmes. | hateful
nid398 | The cultural exchange office has published its annual programme schedule. | unhateful
nid399 | People from that caste should not be allowed to apply for public transport operator licences. | hateful
nid400 | The transport authority has published the application process for operator licences. | unhateful"""

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

with open("scratch/eval_batch4_results.json", "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, indent=2)

print("\n" + "="*50)
print("BATCH 4 EVALUATION COMPLETED (400 SAMPLES):")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Macro F1: {f1_m:.4f}")
print(f"Macro Precision: {prec_m:.4f} | Macro Recall: {rec_m:.4f}")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Total Mismatches: {len(mismatches)}")
print("="*50)


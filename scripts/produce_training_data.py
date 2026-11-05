"""Offline controlled-data construction. No model execution, fitting or training.

Muse supplies and separately reviews scenario wording. Exact outcomes and role
views are constructed locally. Repeated views are correlated, never new roots.
"""
import copy
import json
from pathlib import Path
from fractions import Fraction
from qwen_one_two.training.inputs import canonical, digest

DOMAINS=('refund_eligibility','scheduling','demo_preparation','access_rules','resource_allocation','issue_prioritization')
OPS=('Noul','Choice','Score')
ROLE_RECIPES={
 'reference':'Encode/reconstruct only question, neutral target, alternatives, rubric or key meanings. Current outcome is unavailable. Reason is a separate post-decision role.',
 'receiver':'Assess the supplied request and evidence. Outcomes are targets, never request-encoder inputs.',
 'writer':'Produce an allowed request from the task/evidence. Do not insert a result in its question, target, candidates or rubric.',
 'reader':'Interpret the declared bound return with only request semantics, selected contents and permitted Reason. Original evidence is withheld; invent no cause.',
 'interleaved':'An offline supervised desired trajectory, not an observed rollout: start, selected invocation window, writer request, actual-return requirement, continuation, finish. Runtime producer must use actual model requests/returns and one causal cache.',
 'direct':'Answer the source task directly using supplied facts/rules. Operation invocation is optional, not a requirement.',
 'language_replay':'Grounded native-language preservation proxy: restate supplied policy/facts, extract literal content, format lists, or combine factual notes. No latent operation or budget labels. Synthetic project-owned material, not original pretraining corpus.',
 'decision_replay':'Reuse the reviewed determinate decision exercise as a preservation proxy. Synthetic project-owned material; no claim to recover original Qev training corpus.'}
CORRECTIONS=('question_or_rubric','bank_or_attribute','binding_or_lifecycle','nonassessed_or_contradictory_return','malformed_or_resource')


def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(canonical(value))


def prepare(root):
 root=Path(root)
 pilot=Path('data/pilot-120-20261105/reviewed-scenarios.jsonl')
 existing=[json.loads(l) for l in pilot.read_text().splitlines()]
 # Source/template families are assigned before any instances or views.
 for split in ('train','locked'):
  for domain in DOMAINS:
   count=48
   specs=[{'id':f'{split}-{domain}-{i:02d}','op':OPS[i%3]} for i in range(count)]
   exclusion=[r['human_input']['prompt'] for r in existing if r['domain']==domain]
   prompt=f'''Return exactly48 compact fictional scenario-wording objects for domain {domain}, split {split}. IDs and operations:\n{json.dumps(specs)}\nEach object has exactly id, subject, question, a, b, unit. subject is a short unique concrete task/target (include a distinctive context identity); question is one natural user question relevant to that target. a and b are two DIFFERENT yes/no factual prerequisites, as positive noun phrases, not verdicts. unit is a countable plural unit relevant to the situation (e.g. verified panels, assigned seats). They will supply varied Boolean rules, numeric candidate capacities and fraction rubrics in a deterministic builder. Keep all strings concise. No quantities, answers, operation labels, winner tags, confidence or approvals in the wording. No tools, agents, commentary or markdown; output ONLY a valid JSON array. Aim below6000 output tokens.
Make the48 situations substantively diverse; avoid repeated name/number substitutions. Cover different policies, items, constraints and organizations. No real-world legal/medical advice. All actual rules will be explicitly supplied fictional complete rules, not external knowledge.
This split must use fresh concrete policies/contexts instead of existing pilot scenarios or close paraphrases. Avoid these existing prompts:\n{json.dumps(exclusion)}
For train contexts use invented community/commercial projects. For locked contexts use distinct scientific/cultural fieldwork projects and different policy wording/targets. Do not reuse any supplied task as data. No outcome labels or annotations are requested.''' 
   (root/'prompts'/f'author-{split}-{domain}.txt').write_text(prompt)
 write(root/'split-allocation.json',{'seed':17,'train_root_ids':[f'train-{d}-{i:02d}' for d in DOMAINS for i in range(48)],'locked_root_ids':[f'locked-{d}-{i:02d}' for d in DOMAINS for i in range(48)],'pilot_release_sha256':'177147b46629f7ee2ff39b567fd90543621fba07fed6960bca123b6aa32b4c9d','pilot_use':'development only','roles':ROLE_RECIPES})


def scenario(skin,split,domain,index):
 """Create six evidence-grounded states, independent of author/model verdicts."""
 op=OPS[index%3];base=skin['id'];subject=skin['subject'];a=skin['a'];b=skin['b'];unit=skin['unit']
 offset=10 if split=='train' else 110
 cases=[]
 for k in range(6):
  key=f'{base}-s{k}'; facts=[];reason=None;payload=None;feasibility=None
  if op=='Noul':
   question=skin['question']+' Assess whether the supplied eligibility rule is met.'
   rule=f'For {subject}, eligibility holds if and only if both prerequisites hold: {a}; {b}.'
   meanings={'question':{'proposition':f'The eligibility rule is met for {subject}','criterion':f'Both {a} and {b}'},'target':{'referent':subject}}
   states=[([f'{a}: yes.',f'{b}: yes.'],'Assessed',1),([f'{a}: yes.',f'{b}: no.'],'Assessed',0),([f'{a}: yes.'],'Insufficient',None),([f'At the same assessment instant, register1 says {a}: yes; register2 says {a}: no.',f'{b}: yes.','Neither register has precedence.'],'Conflicting',None),([f'Two distinct targets share the name {subject}: target Red has both prerequisites; target Blue fails both.','The requester does not identify Red or Blue.'],'Unresolved',None),([f'{a}: no.'],'Assessed',0)]
   facts,status,value=states[k]
   if status=='Assessed':payload={'p_true':value};answer=('Yes' if value else 'No')+', under the supplied eligibility rule.'
   elif status=='Insufficient':answer=f'Please confirm whether {b} holds.'
   elif status=='Conflicting':answer=f'The records disagree about {a}; please resolve that discrepancy.'
   else:answer='Which target do you mean: Red or Blue?'
  elif op=='Choice':
   need=offset+5
   question=skin['question']+f' Choose from A,B,C for a requirement of at least {need} {unit}; prefer the greatest capacity among usable options.'
   rule=f'An option is usable if and only if its capacity is at least {need} {unit}. Among usable options, choose greatest capacity; tied maxima are equally acceptable.'
   values=[(need+4,need+2,need-1),(need+4,need+4,need-1),(None,None,need-1),([need+9,need-2],need+3,need-1),(need+4,need+2,need-1),(need-1,need-2,need-3)][k]
   props=[]
   for ident,value in zip('ABC',values):
    if isinstance(value,list):text=f'Two registers at the same instant report option {ident} capacity {value[0]} and {value[1]} {unit}; neither has precedence.'
    elif value is None:text=f'Option {ident} capacity is not supplied.'
    else:text=f'Option {ident} capacity is {value} {unit}.'
    facts.append(text);props.append({'identity':ident,'properties':text})
   meanings={'question':{'proposition':f'Which option to use for {subject}','criterion':f'Capacity at least {need} {unit}, then greatest capacity'},'target':{'referent':subject},'alternatives':props}
   feasibility={ident:('unknown' if v is None or isinstance(v,list) else v>=need) for ident,v in zip('ABC',values)}
   status=('Assessed','Assessed','Insufficient','Conflicting','Unresolved','NoFeasibleChoice')[k]
   if status=='Assessed':
    maximum=max(v for v in values if v>=need);selection=[i for i,v in zip('ABC',values) if v==maximum];payload={'acceptable_selection_ids':selection};answer='Use '+ ' or '.join(selection)+': it has the greatest usable capacity.'
   elif k==2:answer='Please provide the missing capacities for A and B.'
   elif k==3:answer='The two capacity records for A conflict and change the ranking; please resolve them.'
   elif k==4:
    question=skin['question']+' Which option is best: A,B,C?'
    rule=f'All three options are usable for {subject}. Best may mean greatest capacity or lowest fee; no priority between these definitions is specified.'
    facts+=['Option A fee is9 credits; option B fee is1 credit; option C fee is4 credits.']
    meanings['question']['criterion']='Best according to greatest capacity or lowest fee, with no stated priority';feasibility={i:True for i in 'ABC'};answer='Should best mean greatest capacity or lowest fee?'
   else:answer='No listed option meets the minimum capacity requirement.'
  else:
   total=(20 if split=='train' and domain in DOMAINS[:2] else (100 if split=='train' else 200))
   middle=total//2 if total==20 else ((index*7)%(total-1))+1
   done=(0,total,middle,None,[total//4,3*total//4],middle)[k]
   question=skin['question']+' What completion score follows from the supplied counts and rubric?'
   rule=f'Completion score equals completed {unit} divided by required {unit}. Zero means none complete; one means all required units complete.'
   facts=[f'The requirement is {total} {unit}.']
   if isinstance(done,list):facts+=[f'Two registers at the same assessment instant report {done[0]} and {done[1]} completed {unit}; neither source has precedence.']
   elif done is not None:facts+=[f'{done} {unit} are complete.']
   meanings={'question':{'proposition':f'Completion score for {subject}','criterion':'Completed count divided by required count'},'target':{'referent':subject},'rubric':{'dimension':'completion','conditions':f'Completed {unit} divided by required {unit}','direction':'higher','zero':'No required units complete','one':'All required units complete'}}
   status=('Assessed','Assessed','Assessed','Insufficient','Conflicting','Unresolved')[k]
   if status=='Assessed':
    value=float(Fraction(done,total));payload={'score_interval':[value,value]};answer=f'{done}/{total} gives a completion score of {value:g}.'
   elif k==3:answer='How many required units are complete?'
   elif k==4:answer='The simultaneous completion counts conflict; please confirm the correct count.'
   else:
    question=skin['question']+' What quality score should this receive under the supplied policy?'
    rule='Quality is scored from0 to1; quality can mean presentation polish or functional coverage, and the policy does not choose between these dimensions.'
    meanings['question']['criterion']='Quality according to presentation polish or functional coverage with no specified interpretation';meanings['rubric']={'dimension':'quality','conditions':'Presentation polish or functional coverage; interpretation unspecified','direction':'higher','zero':'No quality on the chosen dimension','one':'Full quality on the chosen dimension'}
    facts+=['Presentation polish and functional coverage need not coincide.'];answer='Should the score measure presentation polish or functional coverage?'
  if split=='locked':
   question='Field assessment for '+subject+': '+question
   rule=rule.replace('if and only if','exactly when').replace('eligible','admissible')
  labels={'operation':op,'status':status,'payload':payload,'justification':answer}
  if feasibility is not None:labels['feasibility']=feasibility
  if status!='Assessed':reason={'status':status,'support':answer}
  cases.append({'id':key,'root':base,'split':split,'domain':domain,'operation':op,'human_input':{'prompt':question},'supplied_facts':facts,'supplied_rules':[rule],'semantics':meanings,'labels':labels,'reason':reason,'guidance':{'acceptable_continuations':[answer],'acceptable_routes':[op,'direct'],'direct_answer_allowed':True}})
 return cases


def build_domain(root,split,domain):
 root=Path(root);call=root/'calls'/f'author-{split}-{domain}'
 skins=json.loads((call/'response.json').read_text());assert len(skins)==48
 expected={f'{split}-{domain}-{i:02d}':i for i in range(48)};assert {s['id'] for s in skins}==set(expected)
 cases=[]
 for skin in skins:
  assert set(skin)=={'id','subject','question','a','b','unit'} and all(isinstance(v,str) and v.strip() for v in skin.values())
  cases+=scenario(skin,split,domain,expected[skin['id']])
 write(root/'cases'/f'{split}-{domain}.json',cases)
 visible=[]
 for start in range(0,len(cases),6):
  group=cases[start:start+6]
  base={k:v for k,v in group[0].items() if k not in {'id','root','labels','reason','guidance'}}
  states=[]
  for i,c in enumerate(group):
   state={k:v for k,v in c.items() if k not in {'id','root','labels','reason','guidance'}}
   states.append({'i':i,'overrides':{k:v for k,v in state.items() if v!=base.get(k)}})
  visible.append({'root':group[0]['root'],'base':base,'cases':states})
 # Grouped review covers all deterministic views through the exact recipe and
 # instance IDs; it never receives expected outcomes.
 prompt='''Review all the fictional task instances below, with proposed outcomes hidden. No tools, agents, browsing, markdown or extra calls. The subjects and wording were authored in a separate session; numbers/evidence were instantiated by code. Independently derive status and payload from supplied facts/rules and check question/target/candidate/rubric fidelity. Observe uncertainty relative to the question. A decisive false prerequisite suffices for false despite other missing facts. Conflicting means materially opposite same-instant evidence. Unresolved means interpretation/target is ambiguous, not a generic missing measurement. Choice feasibility is usability, not winning; a lower-ranked usable option remains true. Score is rule-derived, not confidence.
Use EXACT canonical statuses: Assessed, Insufficient, Conflicting, Unresolved, NoFeasibleChoice. NoFeasibleChoice is Choice-only, for every option known unusable, and its payload is null, NEVER an empty assessed selection. For Choice feasibility output EXACTLY an object with keys A,B,C and values true,false or "unknown"; usability is not ranking. Do not output usable_ids.
Output ONLY a JSON array of48 root reviews. Each object: {"root":root ID,"verdict":"approve"|"revise","issues":[concrete content problems],"semantics_judgment":"short specific meaning judgment","view_judgment":"short judgment on permitted role views below","effort_preference":0..1,"effort_reason":"brief preference reason","cases":[{"i":state index0..5,"status":...,"payload":... ,"feasibility":Choice only object,"outcome_judgment":"short rule derivation"}]}.
Include all six instances of each root. Inputs use exact delta encoding: take base, then replace top-level fields in each state overrides; other fields stay unchanged. Instance ID is root plus -s plus its index. This compression does not permit omitting any state or changing any evidence. Payload Assessed Noul={"p_true":0|1}; Choice={"acceptable_selection_ids":[all maximum eligible IDs]}; Score={"score_interval":[value,value]}. Non-Assessed payload null; no invented value. Unsupported/malformed input must revise, not fabricate. Intentional nonassessable tasks are legitimate if meaning is faithful. Propose effort as an experiment preference for concise answers from these simple supplied rules, before model outcomes: consider reasoning depth, comparisons and ambiguity, independently of operation identity. It is not a measured minimum/optimal token budget.
Review ALL the following derived views of every instance. They reuse the EXACT task/evidence/meaning, never change facts or labels. Reference/writer inputs exclude current outcomes. Receiver sees request+facts/rules only. Reader sees just request semantics plus declared bound return/Reason, with original evidence withheld; its target continuation must state the supplied conclusion or ask for supplied missing/conflicting information, never infer hidden causes. Direct/native/interleaved views use permitted source facts. Desired trajectories are offline supervision, not observed model runs. Language replay uses literal restatement/extraction of supplied facts/policy and never injects outcomes. Decision replay reuses assessed outcomes. Corrections are prescribed packet interventions, not actual model failures. Variant source ordering and neutral user wording may vary, but content/rules stay exact. All role views remain in their root family/split. Your view_judgment must assess these constraints for the instances.
ROLE RECIPES:\n'''+json.dumps(ROLE_RECIPES)+'\nINSTANCES:\n'+json.dumps(visible,separators=(',',':'))
 (root/'prompts'/f'review-{split}-{domain}.txt').write_text(prompt)
 print(split,domain,len(cases),'instances; review prompt chars',len(prompt))


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','domain']);p.add_argument('root');p.add_argument('split',nargs='?');p.add_argument('domain',nargs='?');a=p.parse_args()
 if a.action=='prepare':prepare(a.root)
 else:build_domain(a.root,a.split,a.domain)

# These are reviewable annotation views, not encoded tensors or model rollouts.
# Semantic targets stay outside model inputs; the runtime producer is separate.
def source_text(c):
 return c['human_input']['prompt']+'\nFacts:\n'+'\n'.join(c['supplied_facts'])+'\nRules:\n'+'\n'.join(c['supplied_rules'])


def bound_return(c):
 semantics={k:copy.deepcopy(v) for k,v in c['semantics'].items() if k in {'question','target','rubric'}}
 selected=[]
 if c['operation']=='Choice' and c['labels']['status']=='Assessed':
  ids=set(c['labels']['payload']['acceptable_selection_ids'])
  selected=[copy.deepcopy(x) for x in c['semantics']['alternatives'] if x['identity'] in ids]
 return {'request_semantics':semantics,'status':c['labels']['status'],'payload':copy.deepcopy(c['labels']['payload']),'selected_contents':selected,'reason':copy.deepcopy(c.get('reason'))}


def make_view(c,category,key,variant=0):
 from qwen_one_two.training.inputs import Exposure,branch_input
 r=copy.deepcopy(c);r['id']=key;r['source_case']=c['id'];r['family']=c.get('family',c['root']);r['category']=category
 r['variant']=variant;r['source_text']=source_text(c)
 r['labels']['schema_version']='controlled-data-v1'
 r['exposure']={'packet_origin':'reviewed_component','observed_model_rollout':False,'semantic_substitution':False,'gold_return':category=='reader','budget_override':None}
 prompt=c['human_input']['prompt']
 if variant:prompt+='\nPlease give a concise grounded reply.' if variant%2 else '\nPlease state the conclusion and any needed clarification.'
 r['human_input']={'prompt':prompt}
 if category in {'writer','interleaved'}:
  r['model_inputs']=branch_input(Exposure('native_writer'),causal_history=prompt,evidence={'facts':c['supplied_facts'],'rules':c['supplied_rules']},draft={},counters={})
 elif category=='receiver':
  r['model_inputs']=branch_input(Exposure('isolated_receiver',packet_origin='reviewed_component'),request=copy.deepcopy(c['semantics']),evidence={'facts':c['supplied_facts'],'rules':c['supplied_rules']})
 elif category=='reader':
  r['model_inputs']=branch_input(Exposure('isolated_reader',packet_origin='reviewed_component',gold_return=True),bound_return=bound_return(c),permitted_prefix='')
 else:r['model_inputs']={'prompt':prompt,'evidence':{'facts':c['supplied_facts'],'rules':c['supplied_rules']}}
 r['supervision']={'request_semantics':copy.deepcopy(c['semantics']),'result':copy.deepcopy(c['labels']),'reason':copy.deepcopy(c.get('reason')),'continuations':copy.deepcopy(c['guidance']['acceptable_continuations'])}
 if category=='interleaved':
  r['trajectory']={'kind':'offline_supervision','observed':False,'one_prefill':True,'one_cache_lineage':True,'invocation_window':'during_answer' if variant%4==3 else 'after_initial_reasoning','calls':[{'operation':c['operation'],'request_target':copy.deepcopy(c['semantics']),'result_target':copy.deepcopy(c['labels']),'runtime_return':'actual Qev return required; supplied target is supervision only'}],'finish':'normal_answer','native_teacher_forcing':True}
 return r


def reference_view(c,key,role):
 r=make_view(c,'reference',key)
 s=c['semantics']
 if role=='question':meaning=s['question']
 elif role=='target':meaning=s['target']
 elif role=='alternative':meaning=s['alternatives'][0]
 elif role=='key':meaning={'property':('prerequisite satisfaction' if c['operation']=='Noul' else 'counted quantity')+' for '+s['target']['referent'],'unit':'boolean' if c['operation']=='Noul' else 'count'}
 elif role=='reason':meaning=c['reason']
 elif role=='midpoint':meaning='Half of the required units for '+s['target']['referent']+' are complete'
 else:meaning=s['rubric'][role]
 assert meaning is not None
 r['reference_role']=role;r['model_inputs']={'role':role,'meaning':copy.deepcopy(meaning)}
 r['supervision']={'reconstruction':copy.deepcopy(meaning)}
 return r


def language_view(c,key,index):
 r=make_view(c,'language_replay',key)
 mode=index%4
 if mode==0:prompt='Restate this supplied policy exactly, without adding advice:\n'+c['supplied_rules'][0];answer=c['supplied_rules'][0]
 elif mode==1:prompt='List only the supplied observations, one per line:\n'+'\n'.join(c['supplied_facts']);answer='\n'.join(c['supplied_facts'])
 elif mode==2:prompt='Return the first observation exactly, without the policy:\n'+'\n'.join(c['supplied_facts']);answer=c['supplied_facts'][0]
 else:prompt='Put these supplied notes in a JSON array of strings. Preserve their exact wording:\n'+'\n'.join(c['supplied_facts']);answer=json.dumps(c['supplied_facts'],ensure_ascii=False)
 r['human_input']={'prompt':prompt};r['model_inputs']={'prompt':prompt};r['supervision']={'continuations':[answer]}
 r['reuse_rights']='Experiment-generated fictional source wording and project-authored deterministic transformations; no external corpus or original pretraining-data claim.'
 return r


def intervention(c,kind,key,side):
 r=make_view(c,'intervention',key)
 r['intervention']={'category':kind,'side':side,'protocol':'clean_component' if kind in {4,9} else 'offline_semantic_intervention'}
 if kind==4:
  r['model_inputs']={'request':c['semantics'],'evidence':{'facts':c['supplied_facts'],'rules':c['supplied_rules']}}
  r['intervention']['external_hidden_fact']={'second_prerequisite':bool(side)}
 if kind==9:
  r['model_inputs']={'bound_return':bound_return(c),'permitted_prefix':''}
  r['exposure'].update(gold_return=True,branch='isolated_reader',return_substitution=True)
 if kind==10:
  r['model_inputs']['supplied_return']=bound_return(c)
  r['exposure'].update(branch='full_evidence_reader_intervention',return_substitution=True)
 if side==0:return r
 if kind==1:
  r['human_input']['prompt']='Is it FALSE that the supplied eligibility rule is met for this target?'
  r['semantics']['question']['proposition']='Negation of: '+r['semantics']['question']['proposition']
  r['labels']['payload']['p_true']=1-r['labels']['payload']['p_true']
 elif kind==2:
  r['supplied_rules']=[c['supplied_rules'][0].replace('both prerequisites hold','at least one prerequisite holds')]
  r['semantics']['question']['criterion']='At least one prerequisite holds'
  r['labels']['payload']['p_true']=1
 elif kind==3:
  r['supplied_facts']=c['supplied_facts'][:1];r['labels']['status']='Insufficient';r['labels']['payload']=None
  r['reason']={'status':'Insufficient','support':'The second required prerequisite is not supplied.'}
 elif kind==4:
  # This hidden difference is external audit material, never a model input.
  r['intervention']['external_hidden_fact']={'second_prerequisite':True}
 elif kind==5:
  mapping={'A':'X','B':'Y','C':'Z'}
  def ren(text):
   for a,b in mapping.items():text=text.replace('Option '+a,'Option '+b).replace('option '+a,'option '+b)
   return text
  r['human_input']['prompt']=r['human_input']['prompt'].replace('A,B,C','X,Y,Z')
  r['semantics']['alternatives']=[{'identity':mapping[a['identity']],'properties':ren(a['properties'])} for a in reversed(c['semantics']['alternatives'])]
  r['supplied_facts']=[ren(x) for x in reversed(c['supplied_facts'])]
  r['labels']['payload']['acceptable_selection_ids']=[mapping[x] for x in c['labels']['payload']['acceptable_selection_ids']]
  r['labels']['feasibility']={mapping[k]:v for k,v in c['labels']['feasibility'].items()}
 elif kind==6:
  # Replace the base by its already reviewed same-root tied-capacity state.
  raise ValueError('kind6 uses the reviewed paired source instance directly')
 elif kind in {7,8}:
  v=c['labels']['payload']['score_interval'][0]
  if kind==7:
   r['supplied_rules']=['Remaining-fraction score equals one minus the supplied completion fraction. Zero means all complete; one means none complete. Lower is better.']
   r['semantics']['rubric'].update(dimension='remaining fraction',conditions='One minus completed count divided by required count',direction='lower',zero='All required units complete',one='No required units complete')
   value=1-v
  else:
   r['supplied_rules']=['Calibrated progress score equals completed count divided by twice the stated required count. Zero means none complete; one means twice the requirement complete.']
   r['semantics']['rubric'].update(conditions='Completed count divided by twice the stated requirement',zero='No units complete',one='Twice the required units complete');value=v/2
  r['human_input']['prompt']='Compute the score for this target under the explicitly changed rubric.'
  r['semantics']['question']['criterion']=r['supplied_rules'][0]
  r['semantics']['question']['proposition']=('Remaining-fraction score' if kind==7 else 'Calibrated progress score')
  r['labels']['payload']={'score_interval':[value,value]}
 elif kind==9:
  r['model_inputs']={'bound_return':bound_return(c),'permitted_prefix':''}
  r['model_inputs']['bound_return']['payload']['p_true']=1-c['labels']['payload']['p_true']
  r['labels']['payload']=copy.deepcopy(r['model_inputs']['bound_return']['payload'])
 elif kind==10:
  r['supplied_return']=bound_return(c);r['supplied_return']['payload']['p_true']=1-c['labels']['payload']['p_true']
 if kind not in {4,9}:
  r['model_inputs']={'prompt':r['human_input']['prompt'],'request':r['semantics'],'evidence':{'facts':r['supplied_facts'],'rules':r['supplied_rules']}}
 if kind==4:
  # Exactly the permitted source, without external hidden world contents.
  r['model_inputs']={'request':c['semantics'],'evidence':{'facts':c['supplied_facts'],'rules':c['supplied_rules']}}
 if kind==10:r['model_inputs']['supplied_return']=r['supplied_return']
 if kind in {1,2}:r['guidance']['acceptable_continuations']=['Yes, under the stated proposition and rule.' if r['labels']['payload']['p_true'] else 'No, under the stated proposition and rule.']
 if kind==3:r['guidance']['acceptable_continuations']=['Please confirm the missing second prerequisite.']
 if kind in {7,8}:r['guidance']['acceptable_continuations']=[f'The score under the changed rubric is {value:g}.']
 if kind==5:r['guidance']['acceptable_continuations']=['Use '+ ' or '.join(r['labels']['payload']['acceptable_selection_ids'])+' under the supplied capacity rule.']
 if kind==9:r['supervision']={'continuations':['Yes, according to the supplied bound result.' if r['labels']['payload']['p_true'] else 'No, according to the supplied bound result.']}
 elif kind==10:r['supervision']={'continuations':['The supplied result conflicts with the explicit evidence; verify or reconsider the result.']}
 else:r['supervision']={'request_semantics':copy.deepcopy(r['semantics']),'result':copy.deepcopy(r['labels']),'reason':copy.deepcopy(r.get('reason')),'continuations':copy.deepcopy(r['guidance']['acceptable_continuations'])}
 r['guidance']['acceptable_continuations']=copy.deepcopy(r['supervision']['continuations'])
 r['labels']['justification']=r['supervision']['continuations'][0]
 return r


def permuted_choice(c,rotation):
 if c['operation']!='Choice' or not rotation%3:return copy.deepcopy(c)
 r=copy.deepcopy(c);names='ABC';mapping={a:names[(i+rotation)%3] for i,a in enumerate(names)}
 def ren(text):
  for a in names:text=text.replace('Option '+a,'Option @'+a).replace('option '+a,'option @'+a)
  for a,b in mapping.items():text=text.replace('@'+a,b)
  return text.replace('A,B,C',','.join(mapping[a] for a in names))
 r['human_input']['prompt']=ren(r['human_input']['prompt']);r['supplied_facts']=[ren(x) for x in r['supplied_facts']]
 r['semantics']['alternatives']=[{'identity':mapping[a['identity']],'properties':ren(a['properties'])} for a in reversed(r['semantics']['alternatives'])]
 r['labels']['feasibility']={mapping[k]:v for k,v in r['labels']['feasibility'].items()}
 if r['labels']['status']=='Assessed':
  ids=[mapping[x] for x in r['labels']['payload']['acceptable_selection_ids']];r['labels']['payload']['acceptable_selection_ids']=ids
  r['guidance']['acceptable_continuations']=['Use '+ ' or '.join(ids)+' under the supplied capacity criterion.']
 r['view_transform']={'kind':'bijective_candidate_rename_and_order','mapping':mapping}
 return r


def ordered_cases(cases):
 """Visit every root before reusing a root; related states remain one family."""
 groups={}
 for c in cases:groups.setdefault(c.get('root',c['id']),[]).append(c)
 import random
 keys=sorted(groups);random.Random(17).shuffle(keys)
 return [groups[k][(i+cycle)%len(groups[k])] for cycle in range(6) for i,k in enumerate(keys)]


def correction_spec(r,kind):
 snapshot={'run':'offline-spec-'+r['id'],'attempt':1,'sealed':kind in {'binding_or_lifecycle','nonassessed_or_contradictory_return'},'question':copy.deepcopy(r['semantics']['question']),'target':copy.deepcopy(r['semantics']['target']),'bank':copy.deepcopy(r['semantics'].get('alternatives')),'rubric':copy.deepcopy(r['semantics'].get('rubric'))}
 r['correction_specification']={'family':kind,'prescribed_snapshot':snapshot,'snapshot_sha256':digest(canonical(snapshot)),'observed_model_failure':False,'requires_runtime_materialization':True,'valid_next_actions':{'question_or_rubric':['revise unsealed meaning','abort if unable to ground meaning'],'bank_or_attribute':['repair available bank entries','never point to an absent entry'],'binding_or_lifecycle':['reject foreign/stale/consumed return'],'nonassessed_or_contradictory_return':['ask for missing evidence or reconcile conflict','do not substitute the original happy-path answer'],'malformed_or_resource':['abort or terminate under actual resource state','mask numeric outcomes on execution failure']}[kind],'eligible_correction_losses':'only after producer binds this prescribed intervention to its actual packet/snapshot; otherwise masked','native_cancellation_target':None}


def training_views(cases):
 views=[];pool=ordered_cases(cases);byid={c['id']:c for c in cases}
 roles=('question','target','alternative','dimension','conditions','zero','one','midpoint','key','reason')
 for i in range(500):
  role=roles[i%10]
  eligible=[c for c in pool if (role not in {'alternative','dimension','conditions','zero','one','midpoint','reason'} or (role=='alternative' and c['operation']=='Choice') or (role=='reason' and c.get('reason') is not None) or (role in {'dimension','conditions','zero','one','midpoint'} and c['operation']=='Score' and (role!='midpoint' or c['id'].rsplit('-s',1)[1]!='5')))]
  c=eligible[(i//10)%len(eligible)];views.append(reference_view(c,f'train-reference-{i:04d}',role))
 for category,n in [('receiver',1500),('writer',750),('reader',750),('interleaved',2000),('direct',500)]:
  for i in range(n):
   c=pool[i%len(pool)];c=permuted_choice(c,i%3)
   r=make_view(c,category,f'train-{category}-{i:04d}',i%4)
   if category=='interleaved' and i%5==0:
    other=byid[c['root']+'-s'+str((int(c['id'].rsplit('-s',1)[1])+1)%6)]
    other=permuted_choice(other,i%3)
    r['source_case_ids']=[c['id'],other['id']]
    prompt='Assess these two SEPARATE hypothetical situations independently.\nCASE1:\n'+source_text(c)+'\nCASE2:\n'+source_text(other)
    r['human_input']['prompt']=prompt;r['model_inputs']['causal_history']=prompt
    r['model_inputs']['evidence']={'case1':{'facts':c['supplied_facts'],'rules':c['supplied_rules']},'case2':{'facts':other['supplied_facts'],'rules':other['supplied_rules']}}
    request1=copy.deepcopy(c['semantics']);request2=copy.deepcopy(other['semantics'])
    for number,request in enumerate((request1,request2),1):
     request['question']['scope']='CASE'+str(number);request['target']['referent']='CASE'+str(number)+': '+request['target']['referent']
    r['semantics']=request1;r['trajectory']['calls'][0]['request_target']=request1
    r['trajectory']['calls'].append({'operation':other['operation'],'request_target':request2,'result_target':other['labels'],'runtime_return':'actual Qev result required'})
    r['supervision']['request_semantics']=request1;r['supervision']['request_sequence']=[request1,request2];r['supervision']['result_sequence']=[c['labels'],other['labels']]
    r['supervision']['continuations']=['CASE1: '+c['guidance']['acceptable_continuations'][0]+' CASE2: '+other['guidance']['acceptable_continuations'][0]]
    r['source_text']=prompt
   if category=='interleaved' and i%7==0:r['trajectory']['history_use']={'call':0,'reuse':'reuse the same completed bound result later; never consume twice','snapshot':'same request binding'}
   if category in {'writer','reader'} and i<50:correction_spec(r,CORRECTIONS[i%5])
   if category=='interleaved' and i<300:correction_spec(r,CORRECTIONS[i%5])
   views.append(r)
 for i in range(1000):views.append(language_view(pool[i%len(pool)],f'train-language-replay-{i:04d}',i))
 assessed=[c for c in pool if c['labels']['status']=='Assessed']
 for i in range(1000):
  c=permuted_choice(assessed[i%len(assessed)],i%3);r=make_view(c,'decision_replay',f'train-decision-replay-{i:04d}',i//len(assessed))
  r['reuse_rights']='Reviewed experiment-generated fictional decision material reused under its recorded project provenance; no original-Qev corpus claim.'
  views.append(r)
 assert len(views)==8000
 return views


def evaluation_views(cases,split):
 views=[];controlled=[c for c in cases if '-s' in c['id'] and c['id'].rsplit('-s',1)[1].isdigit()];pool=ordered_cases(cases);cpool=ordered_cases(controlled)
 for slice_,count,category in [('RF',100,'reference'),('W',100 if split=='locked' else 70,'writer'),('E',130,'receiver'),('R',130 if split=='locked' else 100,'reader'),('F',1000 if split=='locked' else 400,'direct')]:
  if slice_=='RF':
   roles=('question','target','alternative','dimension','conditions','zero','one','midpoint','key','reason')
   for i in range(count):
    role=roles[i//10]
    eligible=[c for c in cpool if (role in {'question','target','key'} or (role=='alternative' and c['operation']=='Choice') or (role=='reason' and c.get('reason')) or (role in {'dimension','conditions','zero','one','midpoint'} and c['operation']=='Score' and (role!='midpoint' or c['id'].rsplit('-s',1)[1]!='5')))]
    r=reference_view(eligible[i%len(eligible)],f'{split}-RF-{i:04d}',role);r['evaluation_slice']='RF';views.append(r)
  elif slice_ in {'E','R'}:
   cells=[(op,status) for op in OPS for status in (('Assessed','Insufficient','Conflicting','Unresolved','NoFeasibleChoice') if op=='Choice' else ('Assessed','Insufficient','Conflicting','Unresolved'))]
   assigned=[]
   for cell_index,(op,status) in enumerate(cells):
    eligible=[c for c in pool if c['operation']==op and c['labels']['status']==status]
    n=10 if count==130 else 7
    offset=(cell_index%5)*n
    for i in range(n):assigned.append(eligible[(offset+i)%len(eligible)])
   if count==100:
    for op in OPS:assigned += [c for c in cpool if c['operation']==op and c['labels']['status']=='Assessed'][:3]
   for i,c in enumerate(assigned):
    r=make_view(c,category,f'{split}-{slice_}-{i:04d}',i%4);r['evaluation_slice']=slice_;views.append(r)
  elif slice_=='W':
   counts=(34,33,33) if split=='locked' else (24,23,23)
   for op,n in zip(OPS,counts):
    eligible=[c for c in pool if c['operation']==op]
    for i in range(n):
     r=make_view(eligible[i%len(eligible)],category,f'{split}-W-{op}-{i:04d}',i%4);r['evaluation_slice']='W';views.append(r)
  else:
   admissible=[c for c in pool if c.get('effort_admissible',True)]
   for i in range(count):
    r=make_view(admissible[i%len(admissible)],category,f'{split}-F-{i:04d}',i%4);r['evaluation_slice']='F';views.append(r)
 n_pairs=250 if split=='locked' else 100
 for kind in range(1,11):
  op='Choice' if kind in {5,6} else ('Score' if kind in {7,8} else 'Noul')
  state=1 if kind==2 else (2 if kind in {4,7,8} else 0)
  eligible=[c for c in cpool if c['operation']==op and c['id'].endswith('-s'+str(state))]
  for j in range(n_pairs//10):
   c=eligible[((kind-1)*(n_pairs//10)+j)%len(eligible)];pair=f'{split}-X-{kind:02d}-{j:03d}'
   for side in (0,1):
    if kind==6:
     partner=next(x for x in controlled if x['root']==c['root'] and x['id'].endswith('-s'+str(side)))
     r=make_view(partner,'intervention',pair+'-'+str(side))
    else:r=intervention(c,kind,pair+'-'+str(side),side)
    r['evaluation_slice']='X';r.setdefault('intervention',{}).update(category=kind,side=side,pair=pair)
    views.append(r)
 if split=='locked':
  for i in range(40):
   c=cpool[i];r=make_view(c,'mechanical',f'locked-M-{i:04d}');r['labels']={'schema_version':'controlled-data-v1'};r['reason']=None;r['supervision']={'execution_outcome':('reject_foreign_binding','reject_stale_binding','discard_cancelled_result','reject_exhausted_call')[i%4]};r['evaluation_slice']='M';r['mechanical_only']=True;r['model_inputs']=mechanical_fixture(r['id'],i%4);views.append(r)
 assert len(views)==(2000 if split=='locked' else 1000),(split,len(views))
 return views


def mechanical_fixture(key,kind):
 state={'run':key,'snapshot_identity':1,'snapshot_version':2,'cancelled':False,'budget':512}
 completion={'run':key,'snapshot_identity':1,'snapshot_version':2}
 if kind==0:completion['run']='foreign-run'
 elif kind==1:completion['snapshot_version']=1
 elif kind==2:state['cancelled']=True
 else:state['budget']=0
 return {'runtime_state':state,'completion':completion,'fixture_kind':kind}


def verify_mechanical_fixture(data):
 from qwen_one_two.codec.runtime import Runtime,Call
 from qwen_one_two.codec.records import Ref
 from qwen_one_two.codec.schema import Kind,Port,Status,Invalid
 state=data['runtime_state'];completion=data['completion'];kind=data['fixture_kind']
 rt=Runtime(state['run'],10,state['budget'],16);rt.startup()
 if kind==3:
  try:rt.begin(Port.NOUL)
  except Invalid:return True
  return False
 current=Ref(state['run'],Kind.SNAPSHOT,state['snapshot_identity'],state['snapshot_version'])
 rt.call=Call(1,Port.NOUL,rt.position,0,state='Evaluating',snapshot=current);rt.state='Evaluating'
 if state['cancelled']:rt.terminal='Cancelled'
 returned=Ref(completion['run'],Kind.SNAPSHOT,completion['snapshot_identity'],completion['snapshot_version'])
 return rt.prepare_result(returned,Status.ASSESSED) is None

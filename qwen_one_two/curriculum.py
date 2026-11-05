"""Reproducible synthetic internal-call supervision, never optimality labels.

Training records give the operation, candidate boundaries, decision target and
continuation. The production runtime never imports this module or its rules.
"""
import random


def records(seed=42, count=240):
    rng = random.Random(seed)
    examples = []
    greetings = [('Say hello.', 'Hello!'), ('Translate good morning into French. Only the translation.', 'Bonjour.'),
                 ('What is the capital of France?', 'Paris.'), ('Reply with only the word yes.', 'yes'),
                 ('Name a primary color.', 'Red.'), ('Translate cat into French.', 'Chat.'),
                 ('Write a short greeting.', 'Hello!'), ('What is water made of?', 'Hydrogen and oxygen.')]
    statuses = [('approved', True), ('rejected', False), ('accepted', True), ('declined', False)]
    for i in range(count):
        kind = i % 4
        if kind == 0:
            prompt, answer = rng.choice(greetings)
            examples.append(dict(prompt=prompt, prefix='', action=0, target=0., budget=0., answer=answer, options=[]))
        elif kind == 1:
            status, yes = rng.choice(statuses)
            name = rng.choice(['Mira', 'Jon', 'Sana', 'Leon', 'Ari', 'Nora'])
            prompt = f'{name}\'s application was {status}. Is the application approved? Answer yes or no.'
            prefix = rng.choice(['I need to check whether the application was approved.', 'I should make a yes or no decision from the stated status.', 'The stated status determines the answer.'])
            examples.append(dict(prompt=prompt, prefix=prefix, action=1, target=float(yes), budget=.025,
                                 answer='Yes.' if yes else 'No.', options=[]))
        elif kind == 2:
            wanted, other = rng.sample(['red', 'blue', 'green', 'yellow', 'purple', 'orange'], 2)
            opts = [wanted, other]; rng.shuffle(opts)
            prompt = f'The requested color is {wanted}. Choose the matching option.\nA: {opts[0]}\nB: {opts[1]}'
            prefix = rng.choice(['I need to select the option that matches the requested color.', 'I should choose between the listed alternatives.', 'The matching option is the one to select.'])
            examples.append(dict(prompt=prompt, prefix=prefix, action=2, target=opts.index(wanted), budget=.025,
                                 answer=f'{"AB"[opts.index(wanted)]}: {wanted}.', options=opts))
        else:
            value = rng.choice([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
            prompt = f'A progress indicator shows {int(value * 100)} percent complete. Give its completion score between 0 and 1.'
            prefix = rng.choice(['I need a continuous score for the stated completion.', 'I should score completion on the requested zero to one scale.', 'The percentage determines the completion score.'])
            examples.append(dict(prompt=prompt, prefix=prefix, action=3, target=value, budget=.025,
                                 answer=f'{value:g}', options=[]))
    # Broad budget anchors: preferences only, no answer efficacy assessment.
    for value, prompt in [(0.08,'Explain why ice floats.'),(.12,'What is 23 times 19? Show the calculation.'),
                          (.35,'Explain and repair a binary search loop.'),(.7,'Prove there are infinitely many primes.'),
                          (.85,'Derive the edit distance recurrence and prove it.'),(.95,'Design a crash-safe distributed job queue.')]:
        examples.append(dict(prompt=prompt,prefix='',action=0,target=0.,budget=value,answer='',options=[]))
    return examples


def prompt_tokens(tok, prompt):
    return tok.encode(tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=False,
                      add_generation_prompt=True, enable_thinking=True), add_special_tokens=False)


def encode(tok, example):
    ids = prompt_tokens(tok, example['prompt'])
    base_len = len(ids)
    prefix = tok.encode(example['prefix'], add_special_tokens=False)
    # Candidate labels are token endpoints in the user's options, not runtime regexes.
    candidate_positions = []
    for label, value in zip('AB', example['options']):
        marker = tok.encode(f'{label}: {value}', add_special_tokens=False)
        hits = [i + len(marker) - 1 for i in range(base_len-len(marker)+1) if ids[i:i+len(marker)] == marker]
        if len(hits) != 1: raise ValueError('Training candidate token alignment failed')
        candidate_positions.append(hits[0])
    return dict(example, ids=ids, prefix_ids=prefix, candidate_positions=candidate_positions)

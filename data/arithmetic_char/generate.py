import random

# --- 模板库 ---
templates = {
    "simple": [
        # 纯算式
        "{A}{op}{B}={C}",
        "{A} {op} {B} = {C}",
        "{A} {op} {B} -> {C}",
        "{A} {op} {B} => {C}",
        # 基础问答
        "Q:{A}{op}{B}? A:{C}",
        "Question: {A} {op} {B} Answer: {C}",
        # 输入/输出
        "Input: {A} {op} {B} | Output: {C}",
        # 简单陈述
        "The {op_text_noun} of {A} and {B} is {C}.",
        "The result of {A} {op} {B} is {C}.",
        "Compute {A} {op} {B}. Result: {C}.",
        "calc({A},{B},'{op}')={C}",
        "{A} {op_text_verb} {B} equals {C}.",
    ],
    "redundant": [
        "Hello, could you be so kind as to calculate {A} {op} {B} for me? Why, certainly! The answer is {C}.",
        "Let me think for a moment... The problem is {A} {op} {B}. Okay, got it. It must be {C}.",
        "So you want to know the answer to {A} {op} {B}? I can confirm that the final result is indeed {C}.",
        "To solve this, we take the first number, {A}, and apply the operation, which is {op_text_noun}, with the second number, {B}. This procedure yields the value {C}.",
        "I am 100% confident that when you compute {A} {op} {B}, the one and only correct answer is {C}. Don't doubt it.",
        "Today we will learn how to solve {A} {op} {B}. It's very simple. The solution we are looking for is {C}.",
        "As per your request, the following calculation has been performed: {A} {op} {B}. The resulting figure is {C}.",
        "Well, okay, so, like, if you have {A} and you {op_text_verb} {B}, you're gonna get {C}, you know?",
        "I'm not an expert, but I think {A} {op} {B} might be {C}. Let me double check. Yes, it is {C}.",
        "This is a simple arithmetic task. The goal is to evaluate the expression {A} {op} {B}. The expression evaluates to {C}.",
        "Did you ask for {A} {op} {B}? I have processed your request. The answer is {C}.",
    ],
    "structural": [
        "The result is {C}. It was derived by computing {A} {op} {B}.",
        "Let's solve a math problem. The first number is {A}. The second number is {B}. The operation is {op_text_noun}. The final result is {C}.",
        "The fact that {A} {op} {B} equals {C} is fundamental to our understanding.",
        "The number {C} is obtained when {B} is {op_text_passive} {A}.",
        "What do you get if you compute {A} {op} {B}? It's a simple question with a simple answer: {C}.",
        "The calculation of {A} {op} {B} (a trivial task) results in {C}.",
        "Calculation Details: Operand 1: {A}, Operand 2: {B}, Operation: {op}\n, Final Result: {C}.",
        "Let x be {A} and y be {B}. The value of x {op} y is {C}.",
        "From {A} and {B}, by applying the {op_text_noun} operator, we get {C}.",
        "{C} is the number you get if you start with {A} and then {op_text_verb} {B}.",
        "If we were to calculate {A} {op} {B}, the screen would display the number {C}.",
    ],
    "noise": [
        "The sky is blue and grass is green. Anyway, let's get to the point: {A} {op} {B} = {C}.",
        "In building {D1}, on floor {D2}, there are 15 offices. But the math problem is {A} {op} {B}, and the answer is {C}.",
        "Many people mistakenly believe that {A} {op} {B} is {wrong_C}. However, after careful calculation, the true result is {C}.",
        "First, reverse the string 'Hello World'. Just kidding. Let's do math instead: {A} {op} {B} gives us {C}.",
        "John has {A} apples. His friend gives him {B} more apples. Now, John has a total of {C} apples.", # Note: This template only works for addition
        "{A} {op} {B} xyz_123_abc = {C}.",
        "The recipe requires {A} grams of flour and {B} grams of sugar, making a total of {C} grams for the dry ingredients.", # Note: This template only works for addition
        "My shopping list includes: {D1} milks, {D2} eggs, 2 breads. On a separate note, the solution to {A} {op} {B} is {C}.",
        "As stated in document C-{D1}, section {D2}, paragraph 1, the result of {A} {op} {B} is {C}.",
        "/* This is a comment block in a program */ let x = 5; /* Now for the real task */ The {op_text_noun} of {A} and {B} is {C}.",
        "Assertion 1: 1+1=2. Assertion 2: The capital of France is Paris. Assertion 3: {A} {op} {B} is {C}. All these are true.",
    ]
}

def get_operator_text(op):
    """Returns context-appropriate text for an operator."""
    if op == '+':
        return {
            "noun": random.choice(["sum", "addition"]),
            "verb": random.choice(["plus", "added to"]),
            "passive": random.choice(["added to", "combined with"]),
        }
    elif op == '-':
        return {
            "noun": random.choice(["difference", "subtraction"]),
            "verb": random.choice(["minus", "subtracting"]),
            "passive": random.choice(["subtracted from"]),
        }
    return {}

def generate_data_point(max_val=99):
    """Generates a single data point using the template library."""
    
    # 1. Generate core arithmetic components
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    op_choice = random.choice(['+', '-'])
    
    # Ensure subtraction result is not negative for simplicity
    if op_choice == '-' and a < b:
        a, b = b, a # Swap to make a >= b
    
    if op_choice == '+':
        c = a + b
    else: # op_choice == '-'
        c = a - b
        
    # 2. Prepare dynamic values for templates
    op_text = get_operator_text(op_choice)
    placeholders = {
        'A': str(a),
        'B': str(b),
        'C': str(c),
        'op': op_choice,
        'op_text_noun': op_text['noun'],
        'op_text_verb': op_text['verb'],
        'op_text_passive': op_text['passive'],
        # Distractors
        'D1': str(random.randint(1, 10)),
        'D2': str(random.randint(11, 20)),
        'wrong_C': str(c + random.choice([-1, 1, 2, -2])),
    }
    
    # 3. Choose a template category with weights
    # You can adjust these weights to control the data distribution
    category = random.choices(
        list(templates.keys()), 
        weights=[0.25, 0.35, 0.25, 0.15], # Simple, Redundant, Structural, Noise
        k=1
    )[0]
    
    # 4. Select a random template from the chosen category
    template_str = random.choice(templates[category])

    # Special handling for templates that only make sense for one operator
    if "John has" in template_str or "recipe requires" in template_str:
        if op_choice == '-': # Reroll if subtraction is chosen for an addition-only template
            return generate_data_point(max_val)
            
    # 5. Fill the template and return
    return template_str.format(**placeholders)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Generating 15 sample data points ---")
    for i in range(15):
        print(f"{i+1:2d}. {generate_data_point()}")
        
    # You can also generate a large dataset and save to a file
    print("\n--- Generating a dataset of 1000 points to 'dataset.txt' ---")
    with open("/cpfs/user/fengmingquan/nanoGPT/data/arithmetic_char/dataset.txt", "w") as f:
         for _ in range(1000):
             f.write(generate_data_point(max_val=1000) + "\n")
    print("Done.")


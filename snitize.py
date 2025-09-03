text = r"""

```json
[
    {
        "question": "Find the inverse Laplace Transform of $F(s) = \frac{2s+3}{s^2+4s+13}$.",
        "explanation": "The problem requires finding the inverse Laplace Transform of a rational function. The denominator has complex roots, so completing the square and using frequency-shifted sine and cosine transforms is necessary. The numerator must be manipulated to match the required forms.",
        "steps": {
            "1": "Complete the square for the denominator: $s^2+4s+13 = s^2+4s+4+9 = (s+2)^2 + 3^2$.",
            "2": "Rewrite $F(s)$ as $F(s) = \frac{2s+3}{(s+2)^2 + 3^2}$.",
            "3": "Express the numerator in terms of $(s+2)$: $2s+3 = 2(s+2) - 4 + 3 = 2(s+2) - 1$.",
            "4": "Split $F(s)$ into two terms: $F(s) = \frac{2(s+2)}{(s+2)^2 + 3^2} - \frac{1}{(s+2)^2 + 3^2}$.",
            "5": "Adjust the second term for the sine transform: $F(s) = 2\frac{s+2}{(s+2)^2 + 3^2} - \frac{1}{3}\frac{3}{(s+2)^2 + 3^2}$.",
            "6": "Apply the inverse Laplace transform properties: $\mathcal{L}^{-1}\left\{\frac{s-a}{(s-a)^2+b^2}\right\} = e^{at}\cos(bt)u(t)$ and $\mathcal{L}^{-1}\left\{\frac{b}{(s-a)^2+b^2}\right\} = e^{at}\sin(bt)u(t)$.",
            "7": "Identify $a=-2$ and $b=3$.",
            "8": "Combine the results: $f(t) = 2e^{-2t}\cos(3t)u(t) - \frac{1}{3}e^{-2t}\sin(3t)u(t)$."
        },
        "options": [
            "a) $e^{-2t}(2\cos(3t) - \frac{1}{3}\sin(3t))u(t)$",
            "b) $e^{-2t}(2\cos(3t) + \frac{1}{3}\sin(3t))u(t)$",
            "c) $e^{-2t}(\cos(3t) - \sin(3t))u(t)$",
            "d) $e^{-2t}(3\cos(2t) - \frac{1}{2}\sin(2t))u(t)$"
        ],
        "answer": "a) $e^{-2t}(2\cos(3t) - \frac{1}{3}\sin(3t))u(t)$"
    },
    {
        "question": "Find the Laplace Transform of $f(t) = \cos^2(2t)u(t)$.",
        "explanation": "To find the Laplace Transform of $\cos^2(2t)$, a trigonometric identity must be used to simplify the function into terms that have known Laplace Transforms. The identity $\cos^2(\theta) = \frac{1 + \cos(2\theta)}{2}$ is applied.",
        "steps": {
            "1": "Apply the trigonometric identity $\cos^2(2t) = \frac{1 + \cos(4t)}{2}$.",
            "2": "Rewrite the function as $f(t) = \frac{1}{2} + \frac{1}{2}\cos(4t)$ for $t \ge 0$.",
            "3": "Take the Laplace Transform of each term separately: $\mathcal{L}\{f(t)\} = \mathcal{L}\{\frac{1}{2}\} + \mathcal{L}\{\frac{1}{2}\cos(4t)\}$.",
            "4": "Use the linearity property: $\mathcal{L}\{f(t)\} = \frac{1}{2}\mathcal{L}\{1\} + \frac{1}{2}\mathcal{L}\{\cos(4t)\}$.",
            "5": "Recall standard Laplace Transforms: $\mathcal{L}\{1\} = \frac{1}{s}$ and $\mathcal{L}\{\cos(at)\} = \frac{s}{s^2+a^2}$.",
            "6": "Substitute $a=4$: $\mathcal{L}\{\cos(4t)\} = \frac{s}{s^2+16}$.",
            "7": "Combine the results: $F(s) = \frac{1}{2} \cdot \frac{1}{s} + \frac{1}{2} \cdot \frac{s}{s^2+16}$.",
            "8": "Simplify to a single fraction: $F(s) = \frac{1}{2s} + \frac{s}{2(s^2+16)} = \frac{s^2+16 + s^2}{2s(s^2+16)} = \frac{2s^2+16}{2s(s^2+16)} = \frac{s^2+8}{s(s^2+16)}$."
        },
        "options": [
            "a) $\frac{s^2+8}{s(s^2+16)}$",
            "b) $\frac{s^2+4}{s(s^2+4)}$",
            "c) $\frac{s^2+8}{s^2(s^2+16)}$",
            "d) $\frac{s^2+16}{s(s^2+8)}$"
        ],
        "answer": "a) $\frac{s^2+8}{s(s^2+16)}$"
    },
    {
        "question": "Find the inverse Laplace Transform of $F(s) = \frac{d}{ds}\left(\frac{1}{s^2+4}\right)$.",
        "explanation": "This question tests the property of Laplace Transforms relating differentiation in the s-domain to multiplication by $t$ in the time domain. The property is $\mathcal{L}\{t f(t)\} = -\frac{d}{ds}F(s)$.",
        "steps": {
            "1": "Recognize the relationship between $F(s)$ and the derivative of a simpler function $G(s) = \frac{1}{s^2+4}$.",
            "2": "Recall the Laplace Transform property: $\mathcal{L}\{t g(t)\} = -\frac{d}{ds}G(s)$.",
            "3": "This implies that if $F(s) = -\frac{d}{ds}G(s)$, then $f(t) = t g(t)u(t)$.",
            "4": "Find the inverse Laplace Transform of $G(s) = \frac{1}{s^2+4}$.",
            "5": "Rewrite $G(s)$ as $G(s) = \frac{1}{2} \cdot \frac{2}{s^2+2^2}$.",
            "6": "Use the standard transform $\mathcal{L}^{-1}\left\{\frac{b}{s^2+b^2}\right\} = \sin(bt)u(t)$ with $b=2$ to find $g(t) = \frac{1}{2}\sin(2t)u(t)$.",
            "7": "Apply the property: $f(t) = t \cdot g(t)u(t)$.",
            "8": "Substitute $g(t)$: $f(t) = t \left(\frac{1}{2}\sin(2t)u(t)\right) = \frac{t}{2}\sin(2t)u(t)$."
        },
        "options": [
            "a) $\frac{t}{2}\sin(2t)u(t)$",
            "b) $\frac{1}{2}\sin(2t)u(t)$",
            "c) $t\cos(2t)u(t)$",
            "d) $\frac{t}{2}\cos(2t)u(t)$"
        ],
        "answer": "a) $\frac{t}{2}\sin(2t)u(t)$"
    },
    {
        "question": "Find the inverse Laplace Transform of $F(s) = \frac{s+1}{(s+2)^2(s+3)}$.",
        "explanation": "This problem requires Partial Fraction Decomposition (PFD) for a function with a repeated pole at $s=-2$ and a simple pole at $s=-3$. After finding the coefficients, standard inverse Laplace Transforms are applied.",
        "steps": {
            "1": "Set up the PFD: $\frac{s+1}{(s+2)^2(s+3)} = \frac{A}{s+2} + \frac{B}{(s+2)^2} + \frac{C}{s+3}$.",
            "2": "Clear denominators: $s+1 = A(s+2)(s+3) + B(s+3) + C(s+2)^2$.",
            "3": "Solve for coefficients by substituting pole values: $s=-2 \implies B=-1$; $s=-3 \implies C=-2$.",
            "4": "Substitute $B$ and $C$ into the equation for $s=0$: $1 = 6A + 3(-1) + 4(-2) \implies 1 = 6A - 11 \implies A=2$.",
            "5": "Rewrite $F(s)$ with coefficients: $F(s) = \frac{2}{s+2} - \frac{1}{(s+2)^2} - \frac{2}{s+3}$.",
            "6": "Apply inverse Laplace transforms: $\mathcal{L}^{-1}\{\frac{1}{s-a}\} = e^{at}u(t)$ and $\mathcal{L}^{-1}\{\frac{n!}{(s-a)^{n+1}}\} = t^n e^{at}u(t)$.",
            "7": "Calculate individual inverse transforms: $2e^{-2t}u(t)$, $-te^{-2t}u(t)$, and $-2e^{-3t}u(t)$.",
            "8": "Combine terms: $f(t) = (2-t)e^{-2t}u(t) - 2e^{-3t}u(t)$."
        },
        "options": [
            "a) $(2-t)e^{-2t}u(t) - 2e^{-3t}u(t)$",
            "b) $(2+t)e^{-2t}u(t) - 2e^{-3t}u(t)$",
            "c) $(2-t)e^{-2t}u(t) + 2e^{-3t}u(t)$",
            "d) $(1-t)e^{-2t}u(t) - 2e^{-3t}u(t)$"
        ],
        "answer": "a) $(2-t)e^{-2t}u(t) - 2e^{-3t}u(t)$"
    },
    {
        "question": "Find the Laplace Transform of $f(t) = (t-1)e^{-2(t-1)}\cos(3(t-1))u(t-1)$.",
        "explanation": "This question combines the time-shifting property and the frequency-domain differentiation property. First, identify the base function and the shift. Then, find the Laplace Transform of the base function using frequency shifting and differentiation properties, and finally apply the time-shifting property.",
        "steps": {
            "1": "Let $g(\tau) = \tau e^{-2\tau}\cos(3\tau)$ and $a=1$. The function is $f(t) = g(t-1)u(t-1)$.",
            "2": "Use the time-shifting property: $\mathcal{L}\{g(t-a)u(t-a)\} = e^{-as}G(s)$, where $G(s) = \mathcal{L}\{g(t)\}$.",
            "3": "Find $G(s) = \mathcal{L}\{\tau e^{-2\tau}\cos(3\tau)\}$.",
            "4": "Use frequency shifting: $\mathcal{L}\{e^{-2t}\cos(3t)\} = \frac{s+2}{(s+2)^2+3^2}$.",
            "5": "Use differentiation in s-domain: $\mathcal{L}\{t h(t)\} = -\frac{d}{ds}H(s)$. Here $h(t)=\cos(3t)$, $H(s)=\frac{s}{s^2+9}$.",
            "6": "Apply frequency shift to $H(s)$: $\mathcal{L}\{e^{-2t}\cos(3t)\} = \frac{s+2}{(s+2)^2+9}$.",
            "7": "Differentiate this result w.r.t. $s$ and negate: $G(s) = -\frac{d}{ds}\left(\frac{s+2}{(s+2)^2+9}\right) = \frac{(s+2)^2-9}{((s+2)^2+9)^2}$.",
            "8": "Substitute $s+2$ back: $G(s) = \frac{s^2+4s+4-9}{(s^2+4s+4+9)^2} = \frac{s^2+4s-5}{(s^2+4s+13)^2}$.",
            "9": "Apply time shift with $a=1$: $F(s) = e^{-s}G(s) = e^{-s}\frac{s^2+4s-5}{(s^2+4s+13)^2}$."
        },
        "options": [
            "a) $e^{-s}\frac{s^2+4s-5}{(s^2+4s+13)^2}$",
            "b) $e^{-s}\frac{s^2+4s+5}{(s^2+4s+13)^2}$",
            "c) $e^{-s}\frac{s^2+4s-5}{(s^2+4s+9)^2}$",
            "d) $e^{-s}\frac{s^2+4s-5}{(s^2+4s+13)}$"
        ],
        "answer": "a) $e^{-s}\frac{s^2+4s-5}{(s^2+4s+13)^2}$"
    }
]
```

"""


def sanitize(text): 
    """Extract a ```json code block from `text` and return the parsed Python object.

    Strategy:
    - Find the first fenced code block that starts with ```json and ends with ```.
    - Try to parse the inner content with json.loads.
    - On JSON decode error, attempt a couple of safe fallbacks (escape backslashes,
      or use ast.literal_eval after simple true/false/null replacements).
    """
    import re
    import json

    # Find the first ```json ... ``` block
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("No ```json code block found in text")

    code = m.group(1).strip()

    # Try direct JSON parse first
    try:
        return json.loads(code)
    except json.JSONDecodeError:
        # Try escaping single backslashes (common when LaTeX backslashes are present)
        try:
            fixed = code.replace('\\', '\\\\')
            return json.loads(fixed)
        except json.JSONDecodeError:
            # Last resort: try ast.literal_eval after converting JSON literals to Python
            import ast
            pyish = code.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            try:
                return ast.literal_eval(pyish)
            except Exception as e:
                raise ValueError('Failed to parse JSON code block') from e


if __name__ == "__main__":
    # Quick smoke test when run as a script
    import pprint
    try:
        data = sanitize(text)[0]["question"]
        print(data)
    except Exception as e:
        print('Error parsing embedded JSON:', e)
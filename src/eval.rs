use crate::types::Expr;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, PartialEq)]
pub enum EvalResult {
    Err(String),
    Expr(Rc<Expr>),
    Unit,
}

#[derive(Debug)]
pub struct Environment {
    pub contexts: Vec<HashMap<String, (Vec<String>, Rc<Expr>)>>,
}

impl Environment {
    pub fn empty() -> Environment {
        Environment {
            contexts: Vec::new(),
        }
    }

    /// Helper function for tests
    pub fn from_vars(vars: &[(&str, Rc<Expr>)]) -> Environment {
        let mut env = Environment::empty();
        env.push_context();
        vars.iter().for_each(|(name, expr)| {
            let _ = env.add_var(name, expr.clone());
        });
        env
    }

    pub fn default() -> Environment {

        let x = Expr::fnum(1.0);

        let defaults: HashMap<String, (Vec<String>, Rc<Expr>)> = [
            ("False".into(), (Vec::new(), Expr::list(&[]))),
            ("True".into(), (Vec::new(), Expr::list(&[x]))),
            ].iter().cloned().collect();

        Environment {
            contexts: vec![defaults],
        }
    }

    /// Looks up the given symbol in the Environment.
    pub fn lookup(&self, symbol: &str) -> Option<(Vec<String>, Rc<Expr>)> {
        self.contexts
            .iter()
            .rev()
            .find(|ctx| ctx.contains_key(symbol))
            .map(|ctx| ctx.get(symbol))
            .flatten()
            .cloned()
    }

    /// Checks whether the given symbol exists in the Environment.
    pub fn contains_key(&self, symbol: &str) -> bool {
        self.contexts
            .iter()
            .rev()
            .find(|ctx| ctx.contains_key(symbol))
            .is_some()
    }

    /// Pushes a new context on the `contexts` stack.
    pub fn push_context(&mut self) {
        self.contexts.push(HashMap::new());
    }

    /// Pops the last context from the `contexts` stack.
    pub fn pop_context(&mut self) {
        self.contexts.pop();
    }

    /// Adds a variable definition to the Environment
    pub fn add_var(&mut self, var: &str, val: Rc<Expr>) -> Result<(), String> {
        self.contexts
            .last_mut()
            .map_or(
                Err("Environment doesn't have a context to add to.".into()),
                |ctx| { ctx.insert(var.into(), (Vec::new(), val.clone())); Ok(()) },
            )
    }

    /// Adds a function definition to the Environment
    pub fn add_fn(&mut self, name: &str, params: &[String], body: Rc<Expr>) -> Result<(), String> {
        self.contexts.last_mut().map_or(
            Err("Environment does not have a context to add to.".into()),
            |ctx| {
                let param_names: Vec<String> = params.iter().map(|s| s.to_string()).collect();
                ctx.insert(name.to_string(), (param_names, body.clone()));
                Ok(())
            },
        )
    }

    pub fn num_contexts(&self) -> usize {
        self.contexts.len()
    }
}

fn evaluate_symbol(
    expr: Rc<Expr>, 
    sym: &str, 
    args: &[Rc<Expr>],
    env: &mut Environment
) -> EvalResult {
    env.lookup(sym).map_or_else(
            || EvalResult::Expr(expr),
        |(param_names, expression)| {
            if param_names.is_empty() {
                eval(expression.clone(), env)
            }
            else {
                if args.len() != param_names.len() {
                    return EvalResult::Err(format!(
                        "Provided {} arguments but expected {}", 
                        args.len(), 
                        param_names.len()
                    ));
                }
                let mapped_args: Result<Vec<(String, Rc<Expr>)>, String> = args
                    .iter()
                    .zip(param_names)
                    .map(|(expr, name)| match eval(expr.clone(), env) {
                        EvalResult::Expr(e) => Ok((name.to_string(), e.clone())),
                        EvalResult::Err(err) => Err(err),
                        _ => Err("Cannot pass Unit as an argument to a function.".into()),
                    })
                    .collect();

                env.push_context();
                let result = mapped_args.map_or_else(
                    |e| EvalResult::Err(e),
                    |arg_tuples| {
                        arg_tuples.iter().for_each(|(name, expr)| {
                            let _ = env.add_var(name, expr.clone());
                        });
                        eval(expression.clone(), env)
                    },
                );
                env.pop_context();
                result
            }
        }
    )
}

/// Generates the output printed to standard out when the user calls print.
pub fn gen_print_output(expr: Rc<Expr>, env: &mut Environment) -> String {
    match &*expr {
        Expr::Symbol(s) => {
            match env.lookup(&s) {
                None => s.to_string(),
                Some((params, e)) if params.len() == 0 => gen_print_output(e, env),
                _ => format!("<func-object: {}>", s.to_string()),
            }
        },
        Expr::FNum(n) => format!("{}", n),
        Expr::List(vals) => {
            let vals_out: Vec<String> = vals.iter()
                .cloned()
                .map(|x| gen_print_output(x, env))
                .collect();
            format!("({})", vals_out.join(" "))
        }
    }
}

fn add_var_to_env(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    // Incorrectly defined let-statement
    if vals.len() != 2 {
        return EvalResult::Err(
            "Invalid variable definition. Should look like (let someVar someExpr)".into(),
        );
    }
    // Otherwise, let-statement has the correct number of elements
    match (&*vals[0], &vals[1]) {
        (Expr::Symbol(s), e) => {
            match eval(e.clone(), env) {
                EvalResult::Expr(e) => {
                    env.add_var(s, e)
                        .map_or_else(
                            |s| EvalResult::Err(s),
                            |_| EvalResult::Unit,
                        )
                },
                EvalResult::Unit => EvalResult::Err("Cannot assign Unit to a variable".into()),
                err => err
            }
        },
        _ => EvalResult::Err(
            "Second element of variable def must be a symbol and third must be an expression.".into()
        ),
    }
}

// Adds a function to the environment.
fn add_fn_to_env(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.len() != 3 {
        EvalResult::Err("Function definitions must follow the pattern (fn fn-name (arg1 arg2 arg3... argn) <Expr>)".into());
    }
    let fn_name = &*vals[0];
    let p_names = &*vals[1];
    let body = &vals[2];
    match (&*fn_name, p_names, body) {
        (Expr::Symbol(fn_name), Expr::List(params), body) => {
            let ps: Result<Vec<String>, String> = params.iter().cloned().map(|e| {
                if let Expr::Symbol(n) = &*e {
                    Ok(n.to_string())
                }
                else {
                    Err("Function parameters must be symbols.".into())
                }
            })
            .collect();
            ps.map_or_else(
                |err| EvalResult::Err(err),
                |xs| env.add_fn(fn_name, xs.as_slice(), body.clone()).map_or_else(
                    |err| EvalResult::Err(err),
                    |_| EvalResult::Unit
                )
            )
        },
        _ => EvalResult::Err("Function definitions must follow the pattern (fn fn-name (arg1 arg2 arg3... argn) <Expr>)".into()),
    }
}

// Done
fn add_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    // He will implement addition. We need to implement 
    // multiplication, subtraction, logical and, and logical or. Patterns
    // are the same.
    if vals.is_empty() {
        return EvalResult::Err("Must perform addition on at least one number".into());
    }
    // "Implementing addition" video
    // What we need to do:
    // For each expression, evaluate it
        // 1. If expr is fnum, eval returns fnum
        // 2. If variable, eval returns lookup expr (which should be true)
        // 3. If compound expr (i.e. list), eval should return fnum
    // These values are mapped Ok or Err, we want to return sum of Ok's.
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only sum numbers.".into()),
            },
            _ => Err("Failed to eval expr.".into())
        })
        .collect::<Result<Vec<f64>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::fnum(xs.iter().sum()))
    )
}

// Done
fn mul_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.is_empty() {
        return EvalResult::Err("Must perform multiplication on at least one number".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only multiply numbers.".into()),
            },
            _ => Err("Failed to eval expr.".into())
        })
        .collect::<Result<Vec<f64>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::fnum(xs.iter().product()))
    )
}

// Possibly done
fn sub_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.len() < 2  {
        return EvalResult::Err("Must perform subtraction on at least 2 numbers".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only subtract numbers.".into()),
            },
            _ => Err("Failed to eval expr.".into())
        })
        .collect::<Result<Vec<f64>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::fnum(
            xs[1..].iter().fold(xs[0], |acc, x| acc - x)
        ))
    )
}

// Possibly done
fn div_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if vals.len() < 2  {
        return EvalResult::Err("Must perform division on at least 2 numbers".into());
    }
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(*n),
                _ => Err("Can only divide numbers.".into()),
            },
            _ => Err("Failed to eval expr.".into())
        })
        .collect::<Result<Vec<f64>, String>>();
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::fnum(
            xs[1..].iter().fold(xs[0], |acc, x| acc / x)
        ))
    )
}

// Possibly done
fn and_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {

    // Check if vals has at least one expression
    if vals.len() < 2  {
        return EvalResult::Err("Must perform AND operations on at least 2 expressions".into());
    }

    // Collects EvalResults and puts them in a Result vector. Catches anything that isn't an eval result.
    // Evaluates expressions
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(exp) => match &*exp {
                Expr::List(l) if *l == &[Expr::fnum(1.0)] => Ok(true),
                Expr::List(l) if *l == &[] => Ok(false),
                Expr::Symbol(s) if s == &"True".to_string() => Ok(true),
                Expr::Symbol(s) if s == &"False".to_string() => Ok(false),
                Expr::FNum(n) if *n == 0.0 => Ok(false),
                Expr::Symbol(_) => Ok(true),
                Expr::FNum(_) => Ok(true),
                _ => Err(format!("You can't AND {:?}", eval(e.clone(), env)).into()),
            },
            _ => Err("Failed to eval expr.".into())
        })
        .collect::<Result<Vec<bool>, String>>();

    // Returns an error if there was an error; otherwise, checks to see if all items are equal to one another.
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::symbol({
            let y = xs.iter().fold(xs[0], |acc, x| acc && *x); // Checking for AND
            if y {
                "True" // If the result is True, return Symbol("True")
            }
            else {
                "False" // If the result is False, return Symbol("False")
            }
        }),
    ))
}

// Eval has been added.
fn or_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {

    // Check if vals has at least one expression
    if vals.len() < 2  {
        return EvalResult::Err("Must perform OR operations on at least 2 expressions".into());
    }

    // Collects EvalResults and puts them in a Result vector. Catches anything that isn't an eval result.
    // Evaluates expressions
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(exp) => match &*exp {
                Expr::List(l) if *l == &[Expr::fnum(1.0)] => Ok(true),
                Expr::List(l) if *l == &[] => Ok(false),
                Expr::Symbol(s) if s == &"True".to_string() => Ok(true),
                Expr::Symbol(s) if s == &"False".to_string() => Ok(false),
                Expr::FNum(n) if *n == 0.0 => Ok(false),
                Expr::Symbol(_) => Ok(true),
                Expr::FNum(_) => Ok(true),
                _ => Err(format!("You can't OR {:?}", eval(e.clone(), env)).into()),
            },
            _ => Err("Failed to eval expr.".into())
        })
        .collect::<Result<Vec<bool>, String>>();

    // Returns an error if there was an error; otherwise, checks to see if all items are equal to one another.
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::symbol({
            //let first = &xs[0];
            //let y = xs.iter().all(|item| *item & *first); // Checking for OR
            let y = xs.iter().fold(xs[0], |acc, x| acc || *x); // Checking for OR
            if y {
                "True" // If the result is True, return Symbol("True")
            }
            else {
                "False" // If the result is False, return Symbol("False")
            }
        }),
    ))
}

fn not_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    // Check if vals has at least one expression
    if vals.len() < 1 || vals.len() > 1 {
        return EvalResult::Err("Must perform NOT operations on at least one expression".into());
    }

    // Collects EvalResults and puts them in a Result vector. Catches anything that isn't an eval result.
    // Evaluates expressions
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(exp) => match &*exp {
                Expr::List(l) if *l == &[Expr::fnum(1.0)] => Ok(true),
                Expr::List(l) if *l == &[] => Ok(false),
                Expr::Symbol(s) if s == &"True".to_string() => Ok(true),
                Expr::Symbol(s) if s == &"False".to_string() => Ok(false),
                Expr::FNum(n) if *n == 0.0 => Ok(false),
                Expr::Symbol(_) => Ok(true),
                Expr::FNum(_) => Ok(true),
                _ => Err(format!("You can't NOT {:?}", eval(e.clone(), env)).into()),
            },
            _ => Err("Failed to eval expr.".into())
        })
        .collect::<Result<Vec<bool>, String>>();

    // Returns an error if there was an error; otherwise, checks to see if all items are equal to one another.
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::symbol({
            let y = !xs[0]; // Checking for NOT
            if y {
                "True" // If the result is True, return Symbol("True")
            }
            else {
                "False" // If the result is False, return Symbol("False")
            }
        }),
    ))
}

fn equality_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {

    // Check if vals has at least one expression
    if vals.is_empty() {
        return EvalResult::Err("Must perform equality on at least one expression".into());
    }

    // Collects EvalResults and puts them in a Result vector. Catches anything that isn't an eval result.
    // Evaluates expressions
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(Expr::FNum(*n)),
                Expr::Symbol(s) => Ok(Expr::Symbol(s.clone())),
                Expr::List(l) => Ok(Expr::List(l.clone())),
            },
            _ => Err("Failed to eval expr.".into())
        })
        .collect::<Result<Vec<Expr>, String>>();

    // Returns an error if there was an error; otherwise, checks to see if all items are equal to one another.
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::symbol({
            let first = &xs[0];
            let y = xs.iter().all(|item| *item == *first); // Checking for equality
            if y {
                "True" // If all items are equal to one another, return Symbol("True")
            }
            else {
                "False" // If not, return Symbol("False")
            }
        }),
    ))
}

fn inequality_vals(vals: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    // Check if vals has at least one expression
    if vals.is_empty() {
        return EvalResult::Err("Must perform equality on at least one expression".into());
    }

    // Collects EvalResults and puts them in a Result vector. Catches anything that isn't an eval result.
    // Evaluates expressions
    let total = vals.iter()
        .map(|e| match eval(e.clone(), env) {
            EvalResult::Expr(exp) => match &*exp {
                Expr::FNum(n) => Ok(Expr::FNum(*n)),
                Expr::Symbol(s) => Ok(Expr::Symbol(s.clone())),
                Expr::List(l) => Ok(Expr::List(l.clone())),
            },
            _ => Err("Failed to eval expr.".into())
        })
        .collect::<Result<Vec<Expr>, String>>();

    // Returns an error if there was an error; otherwise, checks to see if all items are equal to one another.
    total.map_or_else(
        |err| EvalResult::Err(err),
        |xs| EvalResult::Expr(Expr::symbol({
            let first = &xs[0];
            let y = xs.iter().all(|item| *item == *first); // Checking for equality
            if y {
                "False" // If all items are equal to one another, return Symbol("False")
            }
            else {
                "True" // If not, then it is true that the items are !=. Return Symbol("True")
            }
        }),
    ))
}

pub fn if_then_else(blocks: &[Rc<Expr>], env: &mut Environment) -> EvalResult {
    if blocks.len() != 3 {
        return EvalResult::Err("If expressions must have the format (if (<predicate block>) (<then block>) (<else block>))".into());
    }

    // We know we have 3 blocks - we should evaluate the first block and check if it is ()
    match eval(blocks[0].clone(), env) {
        EvalResult::Expr(expr) => {
            match &* expr {
                Expr::List(vs) if vs.len() == 0 => eval(blocks[2].clone(), env),
                _ => eval(blocks[1].clone(), env),
            }
        },
        EvalResult::Unit => EvalResult::Err("If expression predicates must return an expression.".into()),
        err => err
    }
}

/// Evaluates the given expression.
pub fn eval(e: Rc<Expr>, env: &mut Environment) -> EvalResult {
    match &*e {
        Expr::FNum(_) => EvalResult::Expr(e.clone()),
        Expr::Symbol(s) => evaluate_symbol(e.clone(), s, &[], env),
        Expr::List(vals) => {
            if vals.is_empty() {
                return EvalResult::Expr(Expr::list(&[]));
            }
            let op = &*vals[0];
            match op {
                // Case: (+ <Expr> <Expr> <Expr>)
                Expr::Symbol(s) if s == "+" => add_vals(&vals[1..], env),

                // Case: (* <Expr> <Expr> <Expr>)
                Expr::Symbol(s) if s == "*" => mul_vals(&vals[1..], env),

                // Case: (- <Expr> <Expr> <Expr>)
                Expr::Symbol(s) if s == "-" => sub_vals(&vals[1..], env),

                // Case: (/ <Expr> <Expr> <Expr>)
                Expr::Symbol(s) if s == "/" => div_vals(&vals[1..], env),

                // Case: (and True False True)
                Expr::Symbol(s) if s == "and" => and_vals(&vals[1..], env),

                // Case: (or True False True)
                Expr::Symbol(s) if s == "or" => or_vals(&vals[1..], env),

                // Case: (= x y z)
                Expr::Symbol(s) if s == "=" => equality_vals(&vals[1..], env),

                // Case(!= x y z)
                Expr::Symbol(s) if s == "!=" => inequality_vals(&vals[1..], env),

                // Case: (not True)
                Expr::Symbol(s) if s == "not" => not_vals(&vals[1..], env),

                // Case: (let x <Expr>)
                Expr::Symbol(s) if s == "let" => add_var_to_env(&vals[1..], env),

                // Case: (fn my-func (x1 x2 x3) <Expr>)
                Expr::Symbol(s) if s == "fn" => add_fn_to_env(&vals[1..], env),

                // Case: (print <Expr>)
                Expr::Symbol(s) if s == "print" => {
                    let output: Vec<String> = vals[1..]
                        .iter()
                        .cloned()
                        .map(|expr| gen_print_output(expr, env))
                        .collect();
                    println!("{}", output.join(" "));
                    EvalResult::Unit
                },

                Expr::Symbol(s) if s == "if" => if_then_else(&vals[1..], env),
                Expr::Symbol(s) if env.contains_key(&s) => {
                    evaluate_symbol(e.clone(), s, &vals[1..], env)
                }
                _ => {
                    let res: Result<Vec<Rc<Expr>>, EvalResult> = vals.iter()
                        .cloned()
                        .map(|expr| eval(expr, env))
                        .filter(|x| *x != EvalResult::Unit)
                        .map(|x| if let EvalResult::Expr(expr) = x {
                            Ok(expr)
                        }
                        else {
                            Err(x)
                        })
                        .collect();
                    res.map_or_else(
                        |err| err,
                        |exprs| EvalResult::Expr(Expr::list(&exprs))
                    )
                }
            }
        }
    }
}
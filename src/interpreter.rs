use crate::lex::lex;
use crate::parse::parse;
use crate::eval::{eval, Environment, EvalResult};


// _ => Err(format!("Can only AND Boolean symbols. You tried {:?}", eval(e.clone(), env)).into()),

/// Lexes, parses, and evaluates the given program.
pub fn run_interpreter(program: &str) -> EvalResult {
    match lex(&program) {
        Err(err) => EvalResult::Err(format!("Lex error:  {:?}", err)),
        Ok(tokens) => match parse(&tokens) {
            Err(err) => EvalResult::Err(format!("Parse error: {:?}", err)),
            Ok(expr) => {
                let mut env = Environment::default();
                match eval(expr.clone(), &mut env) {
                    EvalResult::Err(err) => EvalResult::Err(format!("{}", err)),
                    EvalResult::Expr(e) => EvalResult::Expr(e.clone()),
                    EvalResult::Unit => EvalResult::Unit,
                }
            }
        }
    }
}

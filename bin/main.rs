use std::env;
use std::fs;
use mlisp::{eval::EvalResult, interpreter::run_interpreter};
//use mlisp::interpreter::run_interpreter;

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("args: {:?}", args);
    assert!(args.len() > 1, "Must supply a file path.");
    let content = fs::read_to_string(&args[1]).expect("There was an error reading the file");

    println!("read content: {}", content);

    let ans = run_interpreter(&content);
    match ans {
        EvalResult::Err(err) => println!("{}", err),
        _ => {},
    }
}

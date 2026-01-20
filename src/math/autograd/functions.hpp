#pragma once
#ifndef AUTOGRAD_FUNCTIONS_HPP
#define AUTOGRAD_FUNCTIONS_HPP


namespace autograd {


    template <typename ScalarType = float>
    class VectorFunction {
        NodeGenerator node_generator;
        std::vector<ExpressionNode*> eval_ordering;

        using VectorExpression = ExpressionNode*;
        VectorExpression root;

    public:

        void operator= (VectorExpression expr) {
            root = expr;

            // Compute now the topological ordering required for non-recursive
            // evaluation. 
            evaluate_graph_non_recursive_ordering(expr, this->eval_ordering);

        }

        ExpressionNode* expr() {
            return root;
        }

        void operator() (EvalMap<ScalarType>& eval_data, Eigen::Ref<EigVec<ScalarType>> vec) {
            // See the notes  inside apply
            return apply(eval_data, vec);
        }

        void apply(EvalMap<ScalarType>& eval_data, Eigen::Ref<EigVec<ScalarType>> vec) {
            if (root == nullptr) {
                throw std::runtime_error("VectorFunction has no root expression");
            }
            if (root->is_2d()) {
                throw std::runtime_error("VectorFunction cannot be 2 dimensional!");
            }
            // Note: WE CANNOT ASSUME no-aliasing of the data vectors because the expression tree is 
            // really an expression DAG because of memoization!
            // We cache intermediate portions of the graphs to avoid repeated computation of 
            // certain values.
            EvalCache<ScalarType> cache;

            vec = eval_node<ScalarType>(root, eval_ordering, eval_data, cache);
            return;
        }

        // Return a reference to its owned generator. 
        NodeGenerator& generator() {
            return node_generator;
        }

    };
    template <typename ScalarType = float>
    class ScalarFunction {

        using VectorVar = ExpressionNode*;
        using ScalarExpression = ExpressionNode*;
        using VectorExpression = ExpressionNode*;

        // A node generator object is assigned to each function for thread
        // safety and to avoid memory leaks: when the function goes out of
        // scope, all its expression nodes are destroyed. Moreover this can
        // be used to share expression nodes among different expressions for
        // different function entries (
        NodeGenerator node_generator;
        ScalarExpression root;
        std::vector<ExpressionNode*> eval_ordering;

    public:

        ScalarFunction() : root(nullptr) {}

        double operator() (EvalMap<ScalarType>& eval_data) {
            // See the notes  inside apply
            return apply(eval_data);
        }

        double apply(EvalMap<ScalarType>& eval_data) {
            if (root == nullptr) {
                throw std::runtime_error("ScalarFunction has no root expression");
            }
            if (root->is_2d() || root->cols() != 1) {
                throw std::runtime_error("ScalarFunction root must have dimension 1");
            }
            // Note: WE CANNOT ASSUME no-aliasing of the data vectors because the expression tree is 
            // really an expression DAG because of memoization!
            // We cache intermediate portions of the graphs to avoid repeated computation of 
            // certain values.
            EvalCache<ScalarType> cache;

            auto vec_result = eval_node<ScalarType>(root, eval_ordering, eval_data, cache);
            if (vec_result.cols() != 1) {
                throw std::runtime_error("Root evaluation did not produce a scalar, this is abnormal for a scalar function");
            }
            return vec_result(0);
        }

        void flush_alloc_data() {
            node_generator.dealloc_all();
        }

        // Return a reference to its owned generator. 
        NodeGenerator& generator() {
            return node_generator;
        }

        ScalarFunction& operator= (ScalarExpression expr) {
            if (expr != nullptr && expr->is_2d() || expr->cols() != 1)
                throw std::runtime_error("Cannot generate a scalar function from a vector expression!");
            root = expr;
            // This compiles the expression into a well defined evaluation ordering.
            evaluate_graph_non_recursive_ordering(expr, this->eval_ordering);

            // Compute now the topological ordering required for non-recursive
            // evaluation. 
            return *this;
        }

        ExpressionNode* expr() {
            return root;
        }

        void derivative(VectorFunction<ScalarType>& func, VectorVar var, unsigned int optimize_steps = 10) {
            // Compute the derivative of this scalar function with respect to a variable
            Differentiator diff;
            VectorExpression expr = diff.differentiate<ScalarType>(root, var, func.generator());

            func = expr;
        }

    };


    template <typename DataType>
    std::ostream& operator<< (std::ostream& os, ScalarFunction< DataType>& sf) {
        auto* root = sf.expr();
        print_as_binary_tree(root);
        return os;
    }

    template <typename DataType>
    std::ostream& operator<< (std::ostream& os, VectorFunction< DataType>& sf) {
        auto* root = sf.expr();
        print_as_binary_tree(root);
        return os;
    }



}

#endif

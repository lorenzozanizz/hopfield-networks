#pragma once
#ifndef MATH_AUTOGRAD_VARIABLES_HPP
#define MATH_AUTOGRAD_VARIABLES_HPP

// Include the required eigen machinery
#include "../matrix/matrix_ops.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <stack>
#include <set>
#include <map>
#include <memory>

namespace autograd {


    typedef unsigned long dim_t;

    // A set of expression types for the evaluation of the 
    // graph derivatives, each expression type generating
    // its own derivative
    enum class ExpressionType {

        // _ Constants _
        Variable,
        Constant,
        Zero,
        Identity,

        // _ Functions _
        Inversion,
        Add,
        Sub,
        Negation,
        ScalarMultiply,
        ScalarPower,
        VectorScalarMultiply,
        Hadamard,
        Division,

        // _ Domain specific functions _ 
        Exponentiation,
        // Summation of a vector variable excluding an index
        Summation,
        LpNorm,
        LpNormDer,
        SoftMax,
        SoftMaxCrossEntropy,
        // Reserved for future (?) use
        Custom

    };

    // Forward declaration just to be sure
    class ExpressionNode;

    class ExpressionNode {

    public:

        const ExpressionType type;
        // Note the reference to its own type, allowed because its just a pointer
        std::vector<ExpressionNode*> children;
        dim_t row;
        dim_t col;

        // Anonymous union to contain all the node data, this is a 
        // bit corny but reduces storage (which is significant when handling
        // images)
        union {
            float vector_constant;
            unsigned long expr_id;
            unsigned long lp_norm;
            unsigned long pow;
        } expression_data;

        ExpressionNode(ExpressionType t) : type(t) { 
            row = col = 1;
        }

        void set_value(float value) {
            expression_data.vector_constant = value;
        }

        void set_pow(unsigned long p) {
            expression_data.pow = p;
        }

        unsigned long get_pow() const {
            return expression_data.pow;
        }

        float get_constant_value() const {
            return expression_data.vector_constant;
        }

        void set_rows(dim_t r) {
            row = r;
        }

        void set_cols(dim_t c) {
            col = c;
        }

        dim_t rows() const {
            return row;
        }

        dim_t cols() const {
            return col;
        }

        void set_p_norm(unsigned long p) {
            expression_data.lp_norm = p;
        }

        unsigned long get_p_norm() const {
            return expression_data.lp_norm;
        }

        std::vector<ExpressionNode*>& get_children() {
            return children;
        }

        void set_expression_id(unsigned long expr_id) {
            expression_data.expr_id = expr_id;
        }

        ExpressionType get_type() const {
            return type;
        }

        bool is_zero() const {
            return type == ExpressionType::Zero || (type == ExpressionType::Constant && 
                expression_data.vector_constant == 0.0);
        }

        bool is_identity() const {
            return type == ExpressionType::Identity;
        }

        bool is_2d() const {
            return row > 1;
        }

    };

    using VectorVariable = ExpressionNode*;

    // Very simple global owner for all nodes (no deallocation, OK for a toy base)
    class NodeGenerator {


        std::vector<std::unique_ptr<ExpressionNode>> nodes;

    public:

        NodeGenerator() : nodes() { }   

        // Explicitly disallow copies, we are stories std::unique_ptr so
        // compiler forbids us from having copy operators!
        NodeGenerator(const NodeGenerator&) = delete;
        NodeGenerator& operator=(const NodeGenerator) = delete;

        std::vector<std::unique_ptr<ExpressionNode>>& all_nodes() {
            return nodes;
        }

        ExpressionNode* create_vector_variable(dim_t cols, dim_t rows = 1) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Variable)
            );
            auto* node = nodes.back().get();
            node->set_cols(cols);
            if (rows != 1)
                node->set_rows(rows);
            return nodes.back().get();
        }

        ExpressionNode* create_vector_constant(dim_t dimension, float v) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Constant)
            );
            nodes.back()->set_cols(dimension);
            nodes.back()->set_value(v);
            return nodes.back().get();
        }

        ExpressionNode* create_vector_op(ExpressionType type,
            const std::vector<ExpressionNode*>& children, unsigned int cols, unsigned int rows = 1) {

            nodes.emplace_back(
                std::make_unique<ExpressionNode>(type)
            );
            // Handle the case where no children may be present
            if (children.size()) {
                nodes.back()->children = children;
            }
            auto* node = nodes.back().get();
            node->set_cols(cols);
            if (rows != 1)
                node->set_rows(rows);
            return nodes.back().get();
        }

        ExpressionNode* identity(unsigned int cols, unsigned int rows = 1) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Identity)
            );
            auto* ret = nodes.back().get();
            ret->set_cols(cols);
            if (rows != 1)
                ret->set_rows(rows);
            return ret;
        }

        ExpressionNode* zero(unsigned int cols, unsigned int rows = 1) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Zero)
            );
            auto* ret = nodes.back().get();
            ret->set_cols(cols);
            if (rows != 1)
                ret->set_rows(rows);
            return ret;
        }

        ExpressionNode* multiply(ExpressionNode* node, double value) {
            // This always ensure that the scalar in VectorScalarMultiply is the first children.

            auto* val = create_vector_constant(/* dimension */ 1, value);
            if (node->is_2d())
                throw std::runtime_error("Operation not supported!");
            if (node->cols() != 1) {
                return create_vector_op(
                    ExpressionType::VectorScalarMultiply, { val, node }, node->cols());
            }
            else {
                return create_vector_op(ExpressionType::ScalarMultiply, { node, val }, 1);
            }
        }

        ExpressionNode* multiply(double value, ExpressionNode* node) {
            return multiply(node, value);
        }

        ExpressionNode* multiply(ExpressionNode* x, ExpressionNode* y) {
            // This always ensure that the scalar in VectorScalarMultiply is the first children.

            if (x->is_2d() || y->is_2d())
                throw std::runtime_error("Operation not supported!");
            if (x->cols() == 1 && y->cols() == 1)
                return create_vector_op(ExpressionType::ScalarMultiply, { x, y }, 1);
            else if (x->cols() == 1) {
                return create_vector_op(
                    ExpressionType::VectorScalarMultiply, { x, y }, std::max(x->cols(), y->cols()));
            }
            else if (y->cols() == 1) {
                return create_vector_op(
                    ExpressionType::VectorScalarMultiply, { y, x }, std::max(x->cols(), y->cols()));
            }
            else
                throw std::runtime_error("Operation not supported: product by two vectors makes no sense!");
            return nullptr;
        }


        ExpressionNode* create_vector_op(ExpressionType type,
            const std::initializer_list<ExpressionNode*> children, unsigned int cols, unsigned int rows = 1) {
            // Overload to allow more comfortable initializer list expressions
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(type)
            );
            // Handle the case where no children may be present
            auto* ret = nodes.back().get();
            if (children.size()) {
                ret->children = children;
            }
            ret->set_cols(cols);
            if (rows != 1)
                ret->set_rows(rows);
            return ret;
        }

        ExpressionNode* squared_l2_norm(ExpressionNode* expr) {
            auto* lp = lp_norm(expr, 2);
            return pow(lp, 2);
        }

        void dealloc_all() {
            nodes.clear();
        }

        inline ExpressionNode* transpose(ExpressionNode* x) {
            return nullptr;
        }

        inline ExpressionNode* negation(ExpressionNode* x) {
            return create_vector_op(ExpressionType::Negation, { x }, x->cols(), x->rows());
        }

        inline ExpressionNode* exponential(VectorVariable x) {
            return create_vector_op(ExpressionType::Exponentiation, { x }, x->cols(), x->rows());
        }

        inline ExpressionNode* inverse(ExpressionNode* x) {
            return create_vector_op(ExpressionType::Inversion, { x }, x->cols(), x->rows());
        }

        inline ExpressionNode* sum(ExpressionNode* a, ExpressionNode* b) {
            if (a->cols() != b->cols() || a->rows() != b->rows())
                throw std::runtime_error("Cannot sum elements of different size");
            return create_vector_op(ExpressionType::Add, { a, b }, a->cols(), a->rows());
        }

        inline ExpressionNode* sum(ExpressionNode* a, float v) {
            // Create a virtual constant from the value v e.g. create
            // a constant vector (just 1 float is stored, not O(n))
            const auto b = create_vector_constant(a->cols(), v);
            return create_vector_op(ExpressionType::Add, { a, b }, a->cols());
        }

        inline ExpressionNode* sub(ExpressionNode* a, ExpressionNode* b) {
            if (a->cols() != b->cols())
                throw std::runtime_error("Cannot sub elements of different size");
            return create_vector_op(ExpressionType::Sub, { a, b }, a->cols());
        }

        inline ExpressionNode* sub(ExpressionNode* a, float v) {
            // Create a virtual constant from the value v e.g. create
            // a constant vector (just 1 float is stored, not O(n))
            const auto b = create_vector_constant(a->cols(), v);
            return create_vector_op(ExpressionType::Sub, { a, b }, a->cols());
        }

        inline ExpressionNode* prod(ExpressionNode* a, ExpressionNode* b) {
            // Elementwise product!
            if (a->is_2d() || b->is_2d())
                throw std::runtime_error("2d hadamard not supported!");
            if (a->cols() != b->cols())
                throw std::runtime_error("Cannot perform hadamard of different size");
            return create_vector_op(ExpressionType::Hadamard, { a, b }, a->cols());
        }

        inline ExpressionNode* lp_norm(ExpressionNode* x, unsigned int p) {
            auto* node = create_vector_op(ExpressionType::LpNorm, { x }, 1);
            node->set_p_norm(p);
            return node;
        }
        ExpressionNode* smce_logits_true(ExpressionNode* logits, ExpressionNode* tru) {
            auto* node = create_vector_op(ExpressionType::SoftMaxCrossEntropy, { logits, tru }, 1);
            return node;
        }

        ExpressionNode* softmax(ExpressionNode* logits) {
            auto* node = create_vector_op(ExpressionType::SoftMax, { logits }, logits->cols());
            return node;
        }

        inline ExpressionNode* lp_der(ExpressionNode* x, unsigned int p, unsigned int cols) {
            auto* node = create_vector_op(ExpressionType::LpNormDer, { x }, cols);
            node->set_p_norm(p);
            return node;
        }

        inline ExpressionNode* pow(ExpressionNode* x, unsigned int p) {
            if (x->is_2d() || x->cols() != 1)
                throw std::runtime_error("Cannot take power of vector!");
            if (p == 1)
                return x;
            auto* node = create_vector_op(ExpressionType::ScalarPower, { x }, 1);
            node->set_pow(p);
            return node;
        }

    };

    std::string stringify_type(ExpressionType tp) {
        switch (tp) {
        case ExpressionType::Variable: return "Var";
        case ExpressionType::Add: return "Sum";
        case ExpressionType::Sub: return "Sub";
        case ExpressionType::ScalarMultiply: return "ScalarMul";
        case ExpressionType::VectorScalarMultiply: return "VectorScalarMul";
        case ExpressionType::Constant: return "Const";
        case ExpressionType::Zero: return "Zero";
        case ExpressionType::SoftMax: return "SoftMax";
        case ExpressionType::SoftMaxCrossEntropy: return "SoftMaxCrossEntropy";
        case ExpressionType::ScalarPower: return "Pow";
        case ExpressionType::Identity: return "Identity";
        case ExpressionType::LpNorm: return "LpNorm";
        case ExpressionType::LpNormDer: return "LpNormDer";
        }
        return "N/a";
    }

    void print_as_binary_tree(const std::string& prefix, ExpressionNode* node, bool isLeft) {
        if (node != nullptr)
        {
            std::cout << prefix;

            std::cout << (isLeft ? "|--" : "^--");
            std::cout << stringify_type(node->get_type()) << "[" << node << "] D:";
            if (node->is_2d())
                std::cout << "( " << node->rows() << ", " << node->cols() << ")";
            else
                std::cout << node->cols();
            std::cout << std::endl;

            auto& children = node->get_children();
            if (children.size() > 0)
                print_as_binary_tree(prefix + (isLeft ? "|   " : "    "), children[0], true);
            if (children.size() > 1)
                print_as_binary_tree(prefix + (isLeft ? "|   " : "    "), children[1], false);
        }
    }

    void print_as_binary_tree(ExpressionNode* node) {
        print_as_binary_tree("", node, false);
    }


    class Differentiator {

        using NodeCache = std::unordered_map<ExpressionNode*, ExpressionNode*>;
        using VectorVar = ExpressionNode*;

        // Keep a differentiation frame to avoid explicit recursion, compute the
        // derivative of a node when all of its required children are ready (either
        // cached or computed before each node)
        struct DiffFrame {
            ExpressionNode* node;
            std::vector<ExpressionNode*> dchildren;
            bool ready;
        };


    public:
        template <typename ScalarType>
        ExpressionNode* differentiate(
            ExpressionNode* root,
            VectorVar variable,
            NodeGenerator& gen
        ) {
            NodeCache cache;
            return differentiate(root, variable, gen, cache);
        }

        ExpressionNode* differentiate(
            ExpressionNode* root,
            VectorVar var,
            NodeGenerator& gen,
            NodeCache& cache
        ) {
            std::stack<DiffFrame> st;
            // No children yet, root is not ready yet
            st.push({ root, {}, false });

            while (!st.empty()) {
                auto& frame = st.top();
                ExpressionNode* node = frame.node;

                // Cached?
                if (cache.count(node)) {
                    st.pop();
                    continue;
                }

                if (no_children_derivative_required(node->get_type()))
                    frame.ready = true;

                if (!frame.ready) {
                    frame.ready = true;

                    // Push children
                    for (auto* child : node->get_children()) {
                        if (!cache.count(child)) {
                            st.push({ child, {}, false });
                        }
                    }
                }
                else {
                    // Fetch all derivatives from the mapping. 
                    frame.dchildren.clear();
                    for (auto* child : node->get_children())
                        frame.dchildren.push_back(cache[child]);
                    ExpressionNode* dnode = differentiate_node(node, frame.dchildren, var, gen);
                    cache[node] = dnode;
                    st.pop();
                }
            }

            auto* der = cache[root];
            // Perform all possible optimizations on the graph. 
            unsigned int performed_opts;
            do {
                performed_opts = 0;
                optimize(der, gen, performed_opts);
            } while (performed_opts > 0);

            return cache[root];

        }

        ExpressionNode* differentiate_node(ExpressionNode* node, const std::vector<ExpressionNode*>& dchildren,
            VectorVar var, NodeGenerator& gen) {

            if (node->get_type() == ExpressionType::Constant) {
                if (node->cols() == 1)
                    return gen.zero(var->cols());
                else
                    return gen.zero(var->cols(), /* outer dim*/ node->rows());
            }
            else if (node->get_type() == ExpressionType::LpNorm) {
                auto* child = node->get_children()[0];
                return gen.lp_der(child, node->get_p_norm(), var->cols());
            }
            else if (node->get_type() == ExpressionType::LpNormDer)
                throw std::runtime_error("Matrix derivative not supported!");
            else if (node->get_type() == ExpressionType::Variable) {
                if (node == var)
                    return gen.identity(var->cols());
                else return gen.zero(var->cols(), node->cols());
            }
            else if (node->get_type() == ExpressionType::SoftMaxCrossEntropy) {
                auto logits = node->get_children()[0];
                auto ref_val = node->get_children()[1];
                return gen.sub(gen.softmax(logits), ref_val);
            }
            else if (node->get_type() == ExpressionType::ScalarMultiply) {
                auto left_d = dchildren[0];
                auto left = node->get_children()[0];
                auto right_d = dchildren[1];
                auto right = node->get_children()[1];

                return gen.sum(
                    gen.multiply(left_d, right),
                    gen.multiply(left, right_d)
                );
            }
            else if (node->get_type() == ExpressionType::Add) {
                auto left_d = dchildren[0];
                auto right_d = dchildren[1];
                return gen.sum(left_d, right_d);
            }
            else if (node->get_type() == ExpressionType::Sub) {
                auto left_d = dchildren[0];
                auto right_d = dchildren[1];
                return gen.sub(left_d, right_d);
            }
            else if (node->get_type() == ExpressionType::ScalarPower) {
                if (node->get_pow() == 1)
                    return dchildren[0];
                else
                    return gen.multiply(
                        gen.multiply(
                            gen.pow(node->get_children()[0], node->get_pow() - 1),
                            node->get_pow()
                        ), dchildren[0]);
            }
            else throw std::runtime_error("operation not supported!");
            return nullptr;
        }

        static constexpr bool no_children_derivative_required(ExpressionType type) {
            if (type == ExpressionType::Variable || type == ExpressionType::Constant ||
                type == ExpressionType::Identity)
                return true;
            return false;
        }

        struct OptFrame {
            ExpressionNode* node = nullptr;
            size_t next_child = 0;
        };

        ExpressionNode* rewrite(ExpressionNode* node, NodeGenerator& gen, unsigned int& optimizations) {
            if (node == nullptr)
                return node;
            
            auto& children = node->get_children();
            if (node->get_type() == ExpressionType::ScalarMultiply) {
                if (children[0]->is_zero() || children[1]->is_zero())
                    return gen.zero(1);
            }
            if (node->get_type() == ExpressionType::VectorScalarMultiply) {
                if (children[0]->is_zero() || children[1]->is_zero())
                    return gen.zero(children[0]->cols(), children[1]->rows());
            }
            if (node->get_type() == ExpressionType::Add) {
                if (children[0]->is_zero() && children[1]->is_zero())
                    return gen.zero(children[0]->cols(), children[0]->rows());
                if (children[0]->is_zero() && !children[1]->is_zero())
                    return children[1];
                if (!children[0]->is_zero() && children[1]->is_zero())
                    return children[0];
            }
            if (node->get_type() == ExpressionType::Sub) {
                // Sub computes children[0]-children[1]
                if (children[0]->is_zero() && children[1]->is_zero())
                    return gen.zero(children[0]->cols(), children[0]->rows());
                if (children[1]->is_zero() && !children[0]->is_zero())
                    return children[0];
                if (children[0]->is_zero() && !children[1]->is_zero())
                    return gen.negation(children[1]);
            }
            if (node->get_type() == ExpressionType::ScalarPower) {
                if (node->get_pow() == 1)
                    return node->get_children()[0];
            }
            return node;
        }

        ExpressionNode* optimize(ExpressionNode* root, NodeGenerator& gen, unsigned int& optimizations) {

            std::stack<OptFrame> stack;
            stack.push({ root, 0 });

            ExpressionNode* last_result = nullptr;
            unsigned int performed_optimizations = 0;

            while (!stack.empty()) {
                OptFrame& f = stack.top();
                ExpressionNode* node = f.node;

                if (f.next_child < node->get_children().size()) {
                    ExpressionNode* child = node->get_children()[f.next_child];
                    f.next_child++;

                    stack.push({ child, 0 });
                    continue;
                }

                ExpressionNode* new_node = rewrite(node, gen, performed_optimizations);

                stack.pop();

                if (!stack.empty()) {
                    OptFrame& parent = stack.top();
                    parent.node->get_children()[parent.next_child - 1] = new_node;
                }

                last_result = new_node;
            }
            optimizations = performed_optimizations;

            return last_result;
        }
    };



    typedef struct {

    } custom_op_t;

    static std::map<unsigned long, custom_op_t> operation_registry;

    void register_custom_operation(long op_id, void* func, void* par_der) {
        // Register the custom operation on a static private registry
        // operation_registry.insert(op_id, par_der);
    }

    // Alias for convenience template
    template <typename ScalarType> 
    using EigVec = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

    template <typename ScalarType>
    using EvalCache = std::unordered_map<ExpressionNode*, EigVec<ScalarType>>;

    // Raw buffer from eigen vector.
    template <typename ScalarType>
    using EvalMap = std::map<ExpressionNode*, 
        std::reference_wrapper<EigVec<ScalarType>>>;

    template <typename ScalarType>
    // NOTE: we make EXTENSIVE use of compilers RVO return value optimization,
    // this vector is in practice always constructed in the returnee's memory.
    EigVec<ScalarType> eval_node(
        ExpressionNode* root,
        std::vector<ExpressionNode*>& eval_ordering,
        EvalMap<ScalarType>& eval_data,
        EvalCache<ScalarType>& cache
    ) {
       // Surely this will not evaluate to a matrix under normal circumstances.
        EigVec<ScalarType> result(root->cols());
       
        for (ExpressionNode* node : eval_ordering) {
            switch (node->get_type()) {
            case ExpressionType::Variable: {
                // Simply take the memory stored in eval data and copy it into the vector.
                cache[node] = eval_data.at(node).get();
                break;
            }
            case ExpressionType::Constant: {
                // This is a vector constant. 
                EigVec<ScalarType> val(node->cols());
                val.setConstant(node->get_constant_value());
                cache[node] = std::move(val);
                break;
            }
            case ExpressionType::Sub: {
                auto* a = node->get_children()[0];
                auto* b = node->get_children()[1];
                cache[node] = cache[a] - cache[b];
                break;
            }
            case ExpressionType::Add: {
                auto* a = node->get_children()[0];
                auto* b = node->get_children()[1];
                cache[node] = cache[a] + cache[b];
                break;
            }
            case ExpressionType::ScalarMultiply: {
                auto* a = node->get_children()[0];
                auto* b = node->get_children()[1];
                // Scalar multiplication
                EigVec<ScalarType> scalar(1);
                scalar(0) = cache[a](0) * cache[b](0);
                cache[node] = scalar;
                break;
            }
            case ExpressionType::VectorScalarMultiply: {
                // Read the note in generator.multiply, we have that the scalar element
                // is always the first children. 
                auto* scalar = node->get_children()[0];
                auto* b = node->get_children()[1];
                ScalarType scalar_value = static_cast<ScalarType>(cache[scalar](0));
                cache[node] = scalar_value * cache[b];
                break;
            }
            case ExpressionType::SoftMaxCrossEntropy: {
                const ScalarType eps = ScalarType(1e-12);
                auto* logits = node->get_children()[0];
                auto* tru = node->get_children()[1];
                // Evaluate the softmax and then apply the function
                auto logits_val = cache[logits];
                auto true_val = cache[tru];

                EigVec<ScalarType> softmaxed = (logits_val.array() - logits_val.maxCoeff()).exp().matrix();
                softmaxed /= softmaxed.sum();
                const ScalarType scalar_loss = -(true_val.array() * (softmaxed.array() + eps).log()).sum();
                EigVec<ScalarType> scalar(1);
                scalar(0) = scalar_loss;
                cache[node] = scalar;
                break;
            }
            case ExpressionType::SoftMax: {
                auto* logits = node->get_children()[0];
                auto logits_val = cache[logits];
                EigVec<ScalarType> softmaxed = (logits_val.array() - logits_val.maxCoeff()).exp().matrix();
                softmaxed /= softmaxed.sum();
                cache[node] = softmaxed;
                break;
            }
            case ExpressionType::ScalarPower: {
                auto* a = node->get_children()[0];
                EigVec<ScalarType> scalar(1);
                scalar(0) = std::pow(cache[a](0), node->get_pow());
                cache[node] = scalar;
                break;
            }
            case ExpressionType::LpNorm: {
                auto* a = node->get_children()[0];
                EigVec<ScalarType> scalar(1);
                scalar(0) = MathOps::lp_norm(cache[a], node->get_p_norm());
                cache[node] = scalar;
                break;
            }
            case ExpressionType::LpNormDer: {
                auto* a = node->get_children()[0];
                auto norm = MathOps::lp_norm(cache[a], node->get_p_norm());
                auto p = node->get_p_norm();
                if (node->get_p_norm() == 2) {
                    cache[node] = cache[a] / norm;
                }
                else if (node->get_p_norm() == 1) {
                    cache[node] = cache[a].array().sign();
                }
                else 
                    cache[node] = (cache[a].array().abs() / norm).pow(p - 1) * cache[node].array().sign();
                break;
            }
            }
        }
        result = cache[root];
        return result;
    }



    bool no_children_evaluation_required(ExpressionType type) {
        if (type == ExpressionType::Variable)
            return true;
        return false;
    }

    struct EvalFrame {
        ExpressionNode* node;
        bool ready;
    };

    void evaluate_graph_non_recursive_ordering(
        ExpressionNode* root,
        std::vector<ExpressionNode*>& node_ordering
    ) {
        std::stack<EvalFrame> st;
        // No children yet, root is not ready yet
        st.push({ root, false });

        std::set<ExpressionNode*> cache;

        while (!st.empty()) {
            auto& frame = st.top();
            ExpressionNode* node = frame.node;

            if (cache.count(node)) {
                st.pop();
                continue;
            }

            if (no_children_evaluation_required(node->get_type()))
                frame.ready = true;

            if (!frame.ready) {
                frame.ready = true;

                for (auto* child : node->get_children()) {
                    if (!cache.count(child)) {
                        st.push({ child, false });
                    }
                }
            }
            else {
                // Pseudo evaluation of the node.
                node_ordering.push_back(node);
                cache.insert(node);
                st.pop();
            }
        }
        return;
    }

} // ! namespace autograd

#endif //!MATH_AUTOGRAD_VARIABLES_HPP
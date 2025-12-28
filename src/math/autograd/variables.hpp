#pragma once
#ifndef MATH_AUTOGRAD_VARIABLES_HPP
#define MATH_AUTOGRAD_VARIABLES_HPP

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <map>
#include <memory>

namespace autograd {


    typedef unsigned long dim_t;

    // A set of expression types for the evaluation of the 
    // graph derivatives, each expression type generating
    // its own derivative
    enum class ExpressionType {

        // _ Basic math operations _
        Variable,
        Inversion,
        Constant,
        Add,
        Multiply,
        Division,

        // _ Domain specific functions _ 
        Exponentiation,
        Sigmoid,
        Relu,
        Selector,
        // Summation of a vector variable excluding an index
        ExclusiveSummation,
        // Summation of a vector variable including the index
        InclusiveSummation,

        ScalarFunc,
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
        dim_t dim;

        // Anonymous union to contain all the node data, this is a 
        // bit corny but reduces storage (which is significant when handling
        // images)
        union {
            float vector_constant;
            unsigned long expr_id;
        } expression_data;

        ExpressionNode(ExpressionType t) : type(t) { }

        void set_value(float value) {
            expression_data.vector_constant = value;
        }

        void set_var_dimension(dim_t dimension) {
            dim = dimension;
        }

        void set_expression_id(unsigned long expr_id) {
            expression_data.expr_id;
        }

        dim_t dimension() const {
            return dim;
        }

    };

    // Very simple global owner for all nodes (no deallocation, OK for a toy base)
    class NodeGenerator {

        using VectorVariable = ExpressionNode*;

        std::vector<std::unique_ptr<ExpressionNode>> nodes;

    public:

        NodeGenerator() : nodes() { }   

        // Explicitly disallow copies, we are stories std::unique_ptr so
        // compiler forbids us from having copy operators!
        NodeGenerator(const NodeGenerator&) = delete;
        NodeGenerator& operator=(const NodeGenerator) = delete;

        ExpressionNode* create_vector_variable(dim_t dimension) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Variable)
            );
            nodes.back()->set_var_dimension(dimension);
            return nodes.back().get();
        }

        ExpressionNode* create_vector_constant(dim_t dimension, float v) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Constant)
            );
            nodes.back()->set_var_dimension(dimension);
            nodes.back()->set_value(v);
            return nodes.back().get();
        }

        ExpressionNode* create_vector_op(ExpressionType type,
            const std::vector<ExpressionNode*>& children) {

            nodes.emplace_back(
                std::make_unique<ExpressionNode>(type)
            );
            // Handle the case where no children may be present
            if (children.size()) {
                nodes.back()->children = children;
                nodes.back()->set_var_dimension(children[0]->dimension());
            }
            return nodes.back().get();
        }

        void dealloc_all() {
            nodes.clear();
        }

        inline ExpressionNode* exponential(VectorVariable x) {
            return create_vector_op(ExpressionType::Exponentiation, { x });
        }

        inline ExpressionNode* sigmoid(VectorVariable x) {
            return create_vector_op(ExpressionType::Sigmoid, { x });
        }

        inline ExpressionNode* relu(VectorVariable x) {
            return create_vector_op(ExpressionType::Relu, { x });
        }

        inline ExpressionNode* inverse(ExpressionNode* a) {
            return create_vector_op(ExpressionType::Inversion, { a });
        }

        inline ExpressionNode* sum(ExpressionNode* a, ExpressionNode* b) {
            return create_vector_op(ExpressionType::Add, { a, b });
        }

        inline ExpressionNode* sum(ExpressionNode* a, float v) {
            // Create a virtual constant from the value v e.g. create
            // a constant vector (just 1 float is stored, not O(n))
            const auto b = create_vector_constant(a->dimension(), v);
            return create_vector_op(ExpressionType::Add, { a, b });
        }

        inline ExpressionNode* prod(ExpressionNode* a, ExpressionNode* b) {
            return create_vector_op(ExpressionType::Multiply, { a, b });
        }

    };

    class Differentiator {

        using NodeCache = std::unordered_map<ExpressionNode*, ExpressionNode*>;

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
            const dim_t var_index,
            NodeGenerator& gen
        ) {
            NodeCache cache;
            return differentiate(root, var_index, gen, cache);
        }

        ExpressionNode* differentiate(
            ExpressionNode* node,
            const dim_t var_index,
            NodeGenerator& gen,
            NodeCache& cache
        ) {
            const auto it = cache.find(node);
            if (it != cache.end()) return it->second;
           
            return nullptr;
        }
    };


    typedef struct {

    } custom_op_t;

    static std::map<unsigned long, custom_op_t> operation_registry;

    void register_custom_operation(long op_id, void* func, void* par_der) {
        // Register the custom operation on a static private registry
        // operation_registry.insert(op_id, par_der);
    }

    template <typename ScalarType = float>
    class Function {

        const dim_t vec_func_size;
        // A node generator object is assigned to each function for thread
        // safety and to avoid memory leaks: when the function goes out of
        // scope, all its expression nodes are destroyed. Moreover this can
        // be used to share expression nodes among different expressions for
        // different function entries (
        NodeGenerator node_generator;
        std::vector<ExpressionNode*> vec_function;

    public:

        Function(
            dim_t func_size
        ) : vec_func_size(func_size), vec_function(func_size) { }

        void operator() (const ScalarType* raw_data, ScalarType* result) {
            // See the notes  inside apply
            apply(raw_data, result);
        }

        void apply(ScalarType* raw_data, ScalarType* result) {
            // Assume complete data and output indipendence, so that
            // processing for each vector entry can happen in paralell without
            // any problem
        }

        dim_t get_func_size() {
            return vec_func_size;
        }

        void flush_alloc_data() {
            node_generator.dealloc_all();
            vec_function.clear();
        }

        // Return a reference to its owned generator. 
        NodeGenerator& generator() {
            return node_generator;
        }

        void assign_all(ExpressionNode* expression) {
            for (int i = 0; i < vec_func_size; ++i)
                vec_function[i] = expression;
        }

        ExpressionNode*& operator[] (dim_t func_dim) {
            return vec_function[func_dim];
        }

        ExpressionNode* component_expr(dim_t comp_index) {
            return vec_function[comp_index];
        }

        void jacobian(const ScalarType* raw_input, ScalarType* output) {

        }
        
    };



    /* INCOMPLETE: REQUIRES DISCUSSION..
    * 
    // Evaluation of a node given a vectorial input 
    float eval_node(ExpressionNode* expression,
        const float* input,
        std::unordered_map<Node*, float>& expression_cache
    ) {
        // First interrogate the expression cache to see if the value was
        // already cached. Note that caching values may be somewhat expensive...
        auto it = cache.find(n);
        if (it != cache.end()) {
            return it->second;
        }
        // Use an explicit cache to keep intermediate results.
        // Also use an explicit stack to avoid possible overflow for nested expressions
        // (also makes code generation more comfy)

        float result = 0.0f;
        switch (n->type) {
        case Node::Type::VAR: {
            result = x[n->var_index];
            break;
        }
        case Node::Type::CONST: {
            result = n->value;
            break;
        }
        case Node::Type::ADD: {
            float a = eval_node(n->children[0], x, cache);
            float b = eval_node(n->children[1], x, cache);
            result = a + b;
            break;
        }
        case Node::Type::MUL: {
            float a = eval_node(n->children[0], x, cache);
            float b = eval_node(n->children[1], x, cache);
            result = a * b;
            break;
        }
        case Node::Type::EXP: {
            float a = eval_node(n->children[0], x, cache);
            result = std::exp(a);
            break;
        }
        }

        cache[n] = result;
        return result;
    }

    */

} // ! namespace autograd

#endif //!MATH_AUTOGRAD_VARIABLES_HPP
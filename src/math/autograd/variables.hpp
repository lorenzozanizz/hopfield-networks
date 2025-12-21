#pragma once
#ifndef MATH_AUTOGRAD_VARIABLES_HPP
#define MATH_AUTOGRAD_VARIABLES_HPP

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>
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

        // Anonymous union to contain all the node data, this is a 
        // bit corny but reduces storage (which is significant)
        union {
            float vector_constant;
            dim_t dimension;
        } expression_data;

        ExpressionNode(ExpressionType t) : type(t) {}

        void set_value(float value) {
            expression_data.vector_constant = value;
        }

        void set_var_dimension(dim_t dimension) {
            expression_data.dimension = dimension;
        }

        dim_t dimension() const {
            return expression_data.dimension;
        }

    };

    // Very simple global owner for all nodes (no deallocation, OK for a toy base)
    struct NodeGenerator {

        std::vector<std::unique_ptr<ExpressionNode>> nodes;

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

    };

    static NodeGenerator g_node_owner;

    void release_node_ownership() {
        // Explicitly release the ownership of the function nodes whenever
        // the function has exhausted its scope. 
        g_node_owner.dealloc_all();

    }

    ExpressionNode* create_vector_constant(dim_t dimension, float v) {
        return g_node_owner.create_vector_constant(dimension, v);
    }

    ExpressionNode* create_vector_variable(dim_t dimension) {
        return g_node_owner.create_vector_variable(dimension);
    }

    using VectorVariable = ExpressionNode*;

    namespace Ops {

        inline ExpressionNode* inverse(ExpressionNode* a) {
            return g_node_owner.create_vector_op(ExpressionType::Inversion, { a });
        }

        inline ExpressionNode* sum(ExpressionNode* a, ExpressionNode* b) {
            return g_node_owner.create_vector_op(ExpressionType::Add, { a, b });
        }

        inline ExpressionNode* sum(ExpressionNode* a, float v) {
            const auto b = g_node_owner.create_vector_constant(a->dimension(), v);
            return g_node_owner.create_vector_op(ExpressionType::Add, { a, b });
        }

        inline ExpressionNode* prod(ExpressionNode* a, ExpressionNode* b) {
            return g_node_owner.create_vector_op(ExpressionType::Multiply, { a, b });
        }


    }

    namespace Funcs {

        inline ExpressionNode* exponential(VectorVariable x) {
            return g_node_owner.create_vector_op(ExpressionType::Exponentiation, { x });
        }

        inline ExpressionNode* sigmoid(VectorVariable x) {
            return g_node_owner.create_vector_op(ExpressionType::Sigmoid, { x });
        }

        inline ExpressionNode* relu(VectorVariable x) {
            return g_node_owner.create_vector_op(ExpressionType::Relu, { x });
        }
    }

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
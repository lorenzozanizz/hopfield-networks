#pragma once
#ifndef MATH_AUTOGRAD_VARIABLES_HPP
#define MATH_AUTOGRAD_VARIABLES_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <stack>
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
        Dot,
        Transpose,
        Multiply,
        Hadamard,
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

    // Lightweight reference to a matrix laid out in memory
    class MatrixReference {

        void relocate();
    };

    class VectorReference {

        void relocate();
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

        std::vector<ExpressionNode*>& get_children() {
            return children;
        }

        void set_expression_id(unsigned long expr_id) {
            expression_data.expr_id;
        }

        ExpressionType get_type() const {
            return type;
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

        ExpressionNode* create_vector_op(ExpressionType type,
            const std::initializer_list<ExpressionNode*> children) {
            // Overload to allow more comfortable initializer list expressions
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(type)
            );
            // Handle the case where no children may be present
            if (children.size()) {
                nodes.back()->children = children;
                nodes.back()->set_var_dimension((*children.begin())->dimension());
            }
            return nodes.back().get();
        }

        ExpressionNode* squared_norm(ExpressionNode* expr) {
            return this->dot(expr, expr);
        }

        void dealloc_all() {
            nodes.clear();
        }

        inline ExpressionNode* transpose(ExpressionNode* x) {
            return nullptr;
        }

        inline ExpressionNode* dot(ExpressionNode* x, ExpressionNode* y) {
            if (x->dimension() != y->dimension()) {
                throw std::runtime_error("Cannot dot product expressions of different size");
            }
            auto* expr = create_vector_op(ExpressionType::Dot, { x, y });
            expr->set_var_dimension(1);
            return expr;
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

        inline ExpressionNode* matmul(ExpressionNode* a) {
            return create_vector_op(ExpressionType::Multiply, { a });
        }

    };

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
                        else {
                            frame.dchildren.push_back(cache[child]);
                        }
                    }
                }
                else {
                    ExpressionNode* dnode = differentiate_node(node, frame.dchildren, var, gen);
                    cache[node] = dnode;
                    st.pop();
                }
            }

            return cache[root];

        }

        ExpressionNode* differentiate_node(ExpressionNode* node, const std::vector<ExpressionNode*>& dchildren,
            VectorVar var, NodeGenerator& gen) {

            if (node->get_type() == ExpressionType::Variable) {
                return gen.sum(dchildren[0], dchildren[1]);
            }
            else if (node->get_type() == ExpressionType::Dot) {
                auto left_d = dchildren[0];
                auto left = node->get_children()[0];
                auto right_d = dchildren[1];
                auto right = node->get_children()[1];

                return gen.sum( gen.dot(left_d, right) , gen.dot(left, right_d) );
            }
            else if (node->get_type() == ExpressionType::Sigmoid) {

            }
            else if (node->get_type() == ExpressionType::Variable) {

            }
            return nullptr;
        }

        static constexpr bool no_children_derivative_required(ExpressionType type) {
            if (type == ExpressionType::Variable || type == ExpressionType::Constant ||
                type == ExpressionType::Sigmoid)
                return true;
            return false;
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
    class VectorFunction {

        using VectorExpression = ExpressionNode*;

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

    public:

        ScalarFunction() : root(nullptr) { }

        void operator() (const ScalarType* raw_data, ScalarType* result) {
            // See the notes  inside apply
            apply(raw_data, result);
        }

        void apply(ScalarType* raw_data, ScalarType* result) {
            // Assume complete data and output indipendence, so that
            // processing for each vector entry can happen in paralell without
            // any problem
        }

        void flush_alloc_data() {
            node_generator.dealloc_all();
        }

        // Return a reference to its owned generator. 
        NodeGenerator& generator() {
            return node_generator;
        }

        ScalarFunction& operator= (ScalarExpression expr) {
            if (expr != nullptr && expr->dimension() != 1)
                throw std::runtime_error("Cannot generate a scalar function from a vector expression!");
            root = expr;
            return *this;
        }

        ExpressionNode* expr() {
            return root;
        }

        void derivative(VectorFunction<ScalarType>& func, VectorVar var, unsigned int optimize_steps = 10) {
            // Compute the derivative of this scalar function with respect to a variable
            Differentiator diff;
            VectorExpression expr = diff.differentiate( root, var, func.generator() );
            // func = expr;
        }

    };

    std::string stringify_type(ExpressionType tp) {
        switch (tp) {
        case ExpressionType::Variable: return "Var";
        case ExpressionType::Add: return "Sum";
        case ExpressionType::Sigmoid: return "Sigmoid";
        case ExpressionType::Dot: return "Dot";

        }
        return "Boh";
    }

    void printBT(const std::string& prefix, ExpressionNode* node, bool isLeft)
    {
        if (node != nullptr)
        {
            std::cout << prefix;

            std::cout << (isLeft ? "|--" : "^--");
            std::cout << stringify_type(node->get_type()) << "[" <<  node <<"]" << std::endl;

            auto& children = node->get_children();
            if (children.size() > 0)
                printBT(prefix + (isLeft ? "|   " : "    "), children[0], true);
            if (children.size() > 1)
                printBT(prefix + (isLeft ? "|   " : "    "), children[1], false);
        }
    }

    void printBT( ExpressionNode* node)
    {
        printBT("", node, false);
    }

    template <typename DataType>
    std::ostream& operator<< (std::ostream& os, ScalarFunction< DataType>& sf) {
        auto* root = sf.expr();
        printBT(root);
        return os;
    }

} // ! namespace autograd

#endif //!MATH_AUTOGRAD_VARIABLES_HPP
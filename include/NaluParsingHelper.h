// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#ifndef NaluParsingHelper_h
#define NaluParsingHelper_h

// yaml for parsing..
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace sierra {
namespace nalu {

class NaluParsingHelper
{

public:
  static void emit(YAML::Emitter& emout, const YAML::Node& node)
  {
    // recursive depth first
    YAML::NodeType::value type = node.Type();
    std::string out;
    switch (type) {
    case YAML::NodeType::Scalar:
      out = node.as<std::string>();
      emout << out;
      break;
    case YAML::NodeType::Sequence:
      emout << YAML::BeginSeq;
      for (unsigned int i = 0; i < node.size(); ++i) {
        const YAML::Node subnode = node[i];
        emit(emout, subnode);
      }
      emout << YAML::EndSeq;
      break;
    case YAML::NodeType::Map:
      emout << YAML::BeginMap;
      for (YAML::const_iterator i = node.begin(); i != node.end(); ++i) {
        const YAML::Node key = i->first;
        const YAML::Node value = i->second;
        out = key.as<std::string>();
        emout << YAML::Key << out;
        emout << YAML::Value;
        emit(emout, value);
      }
      emout << YAML::EndMap;
      break;
    case YAML::NodeType::Null:
      emout << " (empty) ";
      break;
    default:
      std::cerr << "Warning: emit: unknown/unsupported node type" << std::endl;
      break;
    }
  }

  /// uses Emitter to print node to stream
  static void emit(std::ostream& sout, const YAML::Node& node)
  {
    YAML::Emitter out;
    emit(out, node);
    sout << out.c_str() << std::endl;
  }

  static std::string line_info(const YAML::Node& node)
  {
    std::ostringstream sout;
    sout << "(pos,line,column) = (" << node.Mark().pos << ", "
         << node.Mark().line << ", " << node.Mark().column << ")";
    //	 << "Unknown for now" ;
    return sout.str();
  }

  static std::string info(const YAML::Node& node)
  {
    std::ostringstream sout;
    sout << "Node at " << line_info(node) << " => \n";
    emit(sout, node);
    return sout.str();
  }

  /// just prints nodes depth-first
  static void
  traverse(std::ostream& sout, const YAML::Node& node, unsigned int depth = 0)
  {
    using namespace std;

    // recursive depth first
    YAML::NodeType::value type = node.Type();
    string indent((size_t)depth * 2, ' ');
    string out;
    switch (type) {
    case YAML::NodeType::Scalar:
      out = node.as<std::string>();
      sout << indent << "Scalar: " << out << endl;
      break;
    case YAML::NodeType::Sequence:
      sout << indent << "Sequence:" << endl;
      for (unsigned int i = 0; i < node.size(); ++i) {
        const YAML::Node& subnode = node[i];
        sout << indent << "[" << i << "]:" << endl;
        traverse(sout, subnode, depth + 1);
      }
      break;
    case YAML::NodeType::Map:
      sout << indent << "Map:" << endl;
      for (YAML::const_iterator i = node.begin(); i != node.end(); ++i) {
        const YAML::Node key = i->first;
        const YAML::Node value = i->second;
        out = key.as<std::string>();
        sout << indent << "Key: " << out << endl;
        sout << indent << "Value:" << endl;
        traverse(sout, value, depth + 1);
      }
      break;
    case YAML::NodeType::Null:
      sout << indent << "(empty)" << endl;
      break;
    default:
      cerr << "Warning: traverse: unknown/unsupported node type" << endl;
      break;
    }
  }

  /// returns a vector of nodes that match the given key (depth first traversal)
  static void find_nodes_given_key(
    const std::string& key,
    const YAML::Node& node,
    std::vector<const YAML::Node*>& result)
  {
    // recursive depth first
    YAML::NodeType::value type = node.Type();
    if (type != YAML::NodeType::Scalar && type != YAML::NodeType::Null) {
      if (node[key])
        result.push_back(&node);
    }

    switch (type) {
    case YAML::NodeType::Scalar:
      break;
    case YAML::NodeType::Sequence:
      for (unsigned int i = 0; i < node.size(); ++i) {
        const YAML::Node& subnode = node[i];
        find_nodes_given_key(key, subnode, result);
      }
      break;
    case YAML::NodeType::Map:
      for (YAML::const_iterator i = node.begin(); i != node.end(); ++i) {
        // const YAML::Node & key1   = i.first();
        const YAML::Node value = i->second;
        find_nodes_given_key(key, value, result);
      }
      break;
    case YAML::NodeType::Null:
      break;
    default:
      break;
    }
  }
};

} // namespace nalu
} // namespace sierra
#endif

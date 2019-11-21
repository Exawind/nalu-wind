// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//


#ifndef HexNElementDescription_h
#define HexNElementDescription_h

#include <stddef.h>
#include <map>
#include <memory>
#include <vector>
#include "ElementDescription.h"

namespace sierra {
namespace nalu {

struct HexNElementDescription final: public ElementDescription
{
public:
  HexNElementDescription(std::vector<double> nodeLocs);
private:
  void set_subelement_connectivity();
  std::vector<ordinal_type> edge_node_ordinals();
  void set_edge_node_connectivities();
  std::vector<ordinal_type> face_node_ordinals();
  void set_face_node_connectivities();
  std::vector<ordinal_type> volume_node_ordinals();
  void set_volume_node_connectivities();
  void set_subelement_connectivites();
  void set_side_node_ordinals();

  std::pair<ordinal_type, ordinal_type> get_edge_offsets(
    ordinal_type i,
    ordinal_type j,
    ordinal_type k,
    ordinal_type
    edge_ordinal
  );

  std::pair<ordinal_type, ordinal_type> get_face_offsets(
    ordinal_type i,
    ordinal_type j,
    ordinal_type k,
    ordinal_type face_ordinal
  );
  void set_base_node_maps();
  void set_tensor_product_node_mappings();
  void set_boundary_node_mappings();
  void set_isoparametric_coordinates();
  ordinal_type& nmap(ordinal_type i, ordinal_type j, ordinal_type k ) { return nodeMap.at(i+nodes1D*(j+nodes1D*k)); };
  std::vector<ordinal_type>& inmap(ordinal_type j) { return inverseNodeMap.at(j); };
};

} // namespace nalu
} // namespace Sierra

#endif

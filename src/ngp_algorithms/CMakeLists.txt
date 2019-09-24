add_sources(GlobalSourceList
  # Algorithms
  NodalGradEdgeAlg.C
  NodalGradElemAlg.C
  NodalGradBndryElemAlg.C
  EffDiffFluxCoeffAlg.C
  EffSSTDiffFluxCoeffAlg.C
  EnthalpyEffDiffFluxCoeffAlg.C
  MdotEdgeAlg.C
  TurbViscKsgsAlg.C
  WallFuncGeometryAlg.C
  ABLWallFrictionVelAlg.C
  TKEWallFuncAlg.C

  # Algorithm Drivers
  NgpAlgDriver.C
  NodalGradAlgDriver.C
  )

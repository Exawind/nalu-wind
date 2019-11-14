target_sources(nalu PRIVATE
  # Algorithms
  NodalGradEdgeAlg.C
  NodalGradElemAlg.C
  NodalGradBndryElemAlg.C
  EffDiffFluxCoeffAlg.C
  EffSSTDiffFluxCoeffAlg.C
  EnthalpyEffDiffFluxCoeffAlg.C
  MdotEdgeAlg.C
  TurbViscKsgsAlg.C
  TurbViscSSTAlg.C
  WallFuncGeometryAlg.C
  ABLWallFrictionVelAlg.C
  TKEWallFuncAlg.C
  GeometryInteriorAlg.C
  GeometryBoundaryAlg.C
  SDRLowReWallAlg.C
  SDRWallFuncAlg.C

  # Algorithm Drivers
  NgpAlgDriver.C
  NodalGradAlgDriver.C
  TKEWallFuncAlgDriver.C
  GeometryAlgDriver.C
  WallFricVelAlgDriver.C
  SDRWallFuncAlgDriver.C
  )

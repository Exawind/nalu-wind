realms:
  - name: fluidRealm
    boundary_conditions:
    - wall_boundary_condition: bc_lower
      wall_user_data:
        RANS_abl_bc: yes
        fixed_velocity: 6.6
        fixed_height: 90.0

Run this line in the console so the animation pops out
%matplotlib qt

- Red represents patient zero, who can only infect non-infected individuals.  
- Orange represents infected individuals, assumed unable to reinfect others.  
- Infection probability depends on exposure time, distance, and pulmonary ventilation rate, based on the Wells-Riley model.  
- The simulation involves a 20x20m room with 30 individuals over a 3-hour period.  
- Ventilation rates are set for moderate activity, and the infection radius is fixed at 2m.  
- Social behaviours, such as gathering and movement, are simplified with random velocities and pauses.  



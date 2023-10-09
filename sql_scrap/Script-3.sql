select * from locations l 


select * from causes_of_death cod 
inner join icd_10_codes ic on cod.icd_10_codes_id_fkey = ic.id 
inner join locations l ON cod.locations_id_fkey = l.id 


select l.name as state_name, l.acronym, l.fips, 
cod.rate, cod.deaths, cod.population, cod.start_age, cod.end_age, 
ic.name, ic.common_name, ic.parent, 
ic2.name as parent_name, ic2.common_name as parent_common_name
from causes_of_death cod 
inner join icd_10_codes ic on cod.icd_10_codes_id_fkey = ic.id
left join icd_10_codes ic2 on ic.parent = ic2.id 
inner join locations l ON cod.locations_id_fkey = l.id 
where l.level = 'State'



SELECT id FROM public.locations WHERE acronym = 'AL'


select * from icd_10_codes


select * from causes_of_death cod 

select * from locations l where name = 'New Jersey'



ALTER TABLE causes_of_death DROP CONSTRAINT uniqueness;


ALTER TABLE causes_of_death 
ADD CONSTRAINT uniqueness UNIQUE (icd_10_codes_id_fkey, locations_id_fkey, year, start_age, end_age);
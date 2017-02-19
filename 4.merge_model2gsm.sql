####################################################################################
# Name: merge_model2gsm
# Description: Create an Impala table of mobile atlas from the uploaded csv files
# Input: Mobile atlas table in the folder /user/rata.suwantong/impala_apollo_util/techname_data_mobile_atlas_wodetail/
# Version:
#   2016/11/14 RS: Initial version
#   
####################################################################################
*/

/*Upload Mobile Atlas csv on an Impala's Database Folder (here impala_apollo_util) and Write a Mobile Atlas Impala Table*/

drop table if exists apollo_util.mobile_atlas_gsmarena_pre;
CREATE EXTERNAL TABLE if not exists apollo_util.mobile_atlas_gsmarena_pre
 (
	MODEL STRING,
    VENDOR STRING,
    MODEL_GSM STRING,
    YEAR_RELEASED_GSM INT,
    MONTH_RELEASED_GSM INT,
	TIME_RELEASED_GSM DOUBLE,
	PRICE_RELEASED_GSM INT,
	DIAGONAL_SCREEN_SIZE DOUBLE
 )   
 ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' ESCAPED BY ','
 LOCATION '/user/rata.suwantong/impala_rata2/mobile_atlas_gsmarena/';

 

/*Correct the Mobile Atlas Table*/

drop table if exists apollo_util.mobile_atlas_gsmarena;
create table apollo_util.mobile_atlas_gsmarena row format delimited fields terminated by '\t' as ( select 
	MODEL,
    VENDOR,
    MODEL_GSM,
    YEAR_RELEASED_GSM,
    MONTH_RELEASED_GSM,
	PRICE_RELEASED_GSM,
	DIAGONAL_SCREEN_SIZE 
from apollo_util.mobile_atlas_gsmarena_pre where VENDOR !='VENDOR' );

select * from apollo_util.mobile_atlas_gsmarena limit 20; 
select count(*) from apollo_util.mobile_atlas_gsmarena;
select count(distinct model) from apollo_util.mobile_atlas_gsmarena;

drop table if exists apollo_util.devicelist_idsync_jan17_pre;
CREATE EXTERNAL TABLE if not exists apollo_util.devicelist_idsync_jan17_pre
 (
	MODEL STRING,
	MARKETING_NAME STRING,
    VENDOR STRING,
    DISPLAY_HEIGHT INT,
	DISPLAY_WIDTH INT,
	CAMERA_PIXELS DOUBLE,
    YEAR_RELEASED INT,
	DIAGONAL_SCREEN_SIZE DOUBLE,
	REACH INT
 )   
 ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' ESCAPED BY ','
 LOCATION '/user/rata.suwantong/impala_rata2/devicelist_idsync_jan17/';

 

/*Correct the Mobile Atlas Table*/

drop table if exists apollo_util.devicelist_idsync_jan17;
create table apollo_util.devicelist_idsync_jan17 row format delimited fields terminated by '\t' as ( select distinct MODEL,
	MARKETING_NAME,
    VENDOR,
    DISPLAY_HEIGHT,
	DISPLAY_WIDTH,
	CAMERA_PIXELS,
    YEAR_RELEASED,
	DIAGONAL_SCREEN_SIZE,
	REACH 
from apollo_util.devicelist_idsync_jan17_pre where  model!='MODEL' );

select * from apollo_util.devicelist_idsync_jan17 limit 5;

select count(*) from apollo_util.devicelist_idsync_jan17;
select count(distinct model) from apollo_util.devicelist_idsync_jan17;



drop table if exists apollo_util.techname_matched2atlas_pre;
create table apollo_util.techname_matched2atlas_pre row format delimited fields terminated by '\t' as (
	select model, max(reach) as reach, min(marketing_name) as marketing_name, min(vendor) as vendor, max(price_released) as price_released, max(year_released) as year_released, max(month_released) as month_released, max(diagonal_screen_size) as diagonal_screen_size, max(camera_pixels) as camera_pixels, max(display_height) as display_height, max(display_width) as display_width  
	
	from (SELECT a.*, b.month_released_gsm as MONTH_RELEASED, case when b.price_released_gsm is null or b.price_released_gsm >= 2000 then 0 else b.price_released_gsm end as PRICE_RELEASED 
  FROM apollo_util.devicelist_idsync_jan17 a 
left JOIN  apollo_util.mobile_atlas_gsmarena b 
  ON lcase(a.MODEL) LIKE CONCAT(b.MODEL, '%')) C group by model
);

select * from apollo_util.techname_matched2atlas_pre order by reach desc limit 20;


drop table if exists apollo_util.techname_matched2atlas;
create table apollo_util.techname_matched2atlas row format delimited fields terminated by '\t' as (
  select model, reach, marketing_name, vendor, case when price_released = 0 then null else price_released end as price_released, year_released, month_released, time_released, diagonal_screen_size, camera_pixels, display_height, display_width from 
  (select model, reach, marketing_name, vendor, 
		  case when price_released = 0 and (lcase(model) like '%iris%' or lcase(model) like '%lava%' or lcase(model) like '%smart%' or lcase(model) like '%true%' or lcase(model) like '%dtac%' or lcase(model) like '%joey%' or lcase(model) like '%blade%' or lcase(model) like '%eagle%') then 80
					when price_released = 0 and (lcase(model) like '%i-mobile%' or lcase(model) like '%i-style%') then 130 
					when price_released = 0 and lcase(model) like '%vivo%'  then 340
					when price_released = 0 and lcase(model) like '%asus%'  then 125 
					when price_released = 0 and lcase(model) like '%htc%'  then 470 
					when price_released = 0 and lcase(model) like '%lg%'  then 270 
					when price_released = 0 and lcase(model) like '%huawei%'  then 300 
					when price_released = 0 and vendor ='samsung' then 370 
					when price_released = 0 and (lcase(model) like '%x9009%' ) then 400 
					when price_released = 0 and lcase(model) like '%wiko%'  then 130 
					else price_released end as price_released, 
			year_released, month_released, 
		  case when month_released is null then year_released+0.5 else year_released+month_released/12 end as time_released, diagonal_screen_size, camera_pixels, display_height, display_width 
	from apollo_util.techname_matched2atlas_pre) A ) ;
  
 select * from apollo_util.techname_matched2atlas order by reach desc limit 20;
  
select sum(reach) from apollo_util.techname_matched2atlas where price_released = 0 and lcase(model) not like '%iphone%' and lcase(model) not like '%ipad%';

select sum(reach) from apollo_util.techname_matched2atlas where price_released is not null and lcase(model) not like '%iphone%' and lcase(model) not like '%ipad%';
 
impala-shell -i impala.prd.sg1.tapad.com:21000 -B -o /local/home/rata.suwantong/techname_matched2atlas_pre.csv --output_delimiter=',' -q "select * from apollo_util.techname_matched2atlas where vendor !='apple' order by reach desc"
  
echo $'MODEL, REACH, MARKETING_NAME, VENDOR, PRICE_RELEASED, YEAR_RELEASED, MONTH_RELEASED, TIME_RELEASED, DIAGONAL_SCREEN_SIZE, CAMERA_PIXELS, DISPLAY_HEIGHT, DISPLAY_WIDTH' | cat - techname_matched2atlas_pre.csv > techname_matched2atlas.csv
 



/*Example of command on Unix using to:*/

/*Execute the impala code*/
impala-shell -i impala.prd.sg1.tapad.com:21000 -f 01load_mobileatlas.txt
impala-shell -i impala.prd.sg1.tapad.com:21000 -f 02pantip_sighting_device.txt

drop table if exists sgmt_rules.modellist_201701;
create table sgmt_rules.modellist_201701 row format delimited fields terminated by '\t' as (
	select model, case when marketing_name ='' then lcase(model) else lcase(marketing_name) end as marketing_name, case when vendor ='' and lcase(model) like '%lava%' then 'lava' else lcase(vendor) end as vendor, case when display_height= 0 then null else  display_height end as  display_height, case when display_width= 0 then null else  display_width end as  display_width, case when camera_pixels= 0 then null else  camera_pixels end as camera_pixels , case when year_released= 0 then null else year_released end as year_released, 
	case when diagonal_screen_size= 0 then null else diagonal_screen_size end as diagonal_screen_size, reach from apollo_util.devicelist_idsync where lcase(model) not like '%windows%' and lcase(model) not like '%ios%' and lcase(model) not like '%firefox%' and lcase(model) not like '%linux%' and lcase(model) not like '%safari%' and lcase(model) not like '%chrome%' and model != '' and lcase(model) not like '%android%' and char_length(model)>2 order by reach desc
	);
	
	
select * from sgmt_rules.modellist_201701 limit 20;

impala-shell -i impala.prd.sg1.tapad.com:21000 -B -o /local/home/rata.suwantong/devicelist_idsync_jan17_pre.csv --output_delimiter=',' -q "select * from sgmt_rules.modellist_201701 order by reach desc"


/*Name the columns in the csv files */  
echo $'MODEL, MARKETING_NAME, VENDOR, DISPLAY_HEIGHT, DISPLAY_WIDTH, CAMERA_PIXELS, YEAR_RELEASED, DIAGONAL_SCREEN_SIZE, REACH' | cat - devicelist_idsync_jan17_pre.csv > devicelist_idsync_jan17.csv

/* Zip the csv files*/ 
zip -r devicelist_idsync_jan17.zip devicelist_idsync_jan17.csv
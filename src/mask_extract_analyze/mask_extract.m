close all
clear all

% input: folder, im_name, label_name, dataset


dataset='acre';
if strcmp(dataset,'acre')
    folder='/home/lvc/Jorg/igarss/wildfire_fcn/data/AP2_Acre/';
    im_name='L8_002-67_ROI.tif';
    label_name='labels.tif';
elseif strcmp(dataset,'para')
    folder='/home/lvc/Jorg/igarss/wildfire_fcn/data/AP1_Para/';
    label_name='labels.tif';
    im_name='L8_224-66_ROI_clip.tif';
end
im=imread(strcat(folder,im_name));
if strcmp(dataset,'para')
    im=im(1:end-1,1:end-1,:); 
elseif strcmp(dataset,'acre')
    im=im(1:end-1,:,:); 
end

label=imread(strcat(folder,label_name));
label(label==2)=1;

figure();imshow(label,[])

%%

mask=zeros(size(label));
mask(2500:3518,1810:3621)=1;
%mask=1-mask;
mask(1:1200,1:1810)=2;

figure();imshow(mask,[])

reference_masked=label(mask==1);
sum(reference_masked==0)

sum(reference_masked==1)
imshow(mask,[])
imwrite(uint8(mask),'train_test_mask_ac_target.png')
%%

mask_evaluate(im,label,mask)
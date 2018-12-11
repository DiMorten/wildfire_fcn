close all
clear all

% input: folder, im_name, label_name, dataset


%dataset='acre';
dataset='para';
dataset='area23';

if strcmp(dataset,'para') ||strcmp(dataset,'acre') 
    application='wildfire';
elseif strcmp(dataset,'area3') ||strcmp(dataset,'area23') 
    application='vaihinghen';
end

mask_name='TrainTestMask.png';
if strcmp(dataset,'acre')
    folder='/home/lvc/Jorg/igarss/wildfire_fcn/data/AP2_Acre/';
    im_name='L8_002-67_ROI.tif';
    label_name='labels.tif';
elseif strcmp(dataset,'para')
    folder='/home/lvc/Jorg/igarss/wildfire_fcn/data/AP1_Para/';
    label_name='labels.tif';
    im_name='L8_224-66_ROI_clip.tif';
elseif strcmp(dataset,'area3')
    folder='/home/lvc/Jorg/igarss/wildfire_fcn/data/vaihinghen/area3/';
    label_name='labels.tif';
    im_name='im.tif';
elseif strcmp(dataset,'area23')
    folder='/home/lvc/Jorg/igarss/wildfire_fcn/data/vaihinghen/area23/';
    label_name='labels.tif';
    im_name='im.tif';
end
im=imread(strcat(folder,im_name));
if strcmp(dataset,'para')
    im=im(1:end-1,1:end-1,:); 
elseif strcmp(dataset,'acre')
    im=im(1:end-1,:,:); 
end

label=imread(strcat(folder,label_name));
if strcmp(application,'wildfire')
    label(label==2)=1;
    figure();imshow(uint16(im(:,:,1:3))*50,[])
end
figure();
ha(1)=subplot(1,2,1);
imshow(label,[])
%%
% mask=imread(strcat(folder,mask_name));
% subplot(1,2,2)
% 
% imshow(mask*100,[])
% mask=rgb2gray(mask);
% 
% mask_evaluate2(label,mask)
% %mask_evaluate(im,label,mask)
% % mask is 1 for training , 2 for testing, 3 for validation
% % In source, 1+2 are used for training.
% % In target, 2 is used for testing.

%%
if strcmp(dataset,'para')

    mode=2
    if mode==1
        mask=zeros(size(label));
        mask(2500:3518,1810:3621)=1;
        %mask=1-mask;
        mask(1:1200,1:1810)=2;
    elseif mode==2
        mask=ones(size(label))*2;
        mask(1417:1765,700:1281)=1;
        mask(3347:4705,1:699)=1;
        mask(3330:3838,1720:2157)=1;
        mask(2266:2525,1755:1960)=1;
        
        mask(3792:3955,379:497)=3;
        mask(3382:3449,1918:2037)=3;
        mask(4250:4504,531:699)=3;
        mask(1417:1539,1100:1162)=3;
        
        %mask(2266:2268,1755:1756)=3;
    end
elseif strcmp(dataset,'acre')
    mask=ones(size(label))*2;%test
    mask(2933:3362,2301:2877)=1;
    mask(479:1364,222:872)=1;
    mask(1967:2890,2993:3621)=1;
    %mask(2403:2963,481:749)=1;
    mask(2824:3082,1000:1294)=1;
    mask(2781,1)=1;
    
    % Val
    
    mask(1977:2229,3405:3617)=3;
    mask(3088:3311,2499:2624)=3;
    mask(984:1247,250:393)=3;
    
elseif strcmp(dataset,'area3')
    mask=ones(size(label))*2;%test
    mask(511:609,371:530)=3;
elseif strcmp(dataset,'area23')
    mask=ones(size(label))*2;%test
    mask(2102:2251,1294:1434)=3;
end
figure(2);
ha(2)=subplot(1,2,2);
 
imshow(mask*100,[])
%imshow(mask,[])

linkaxes(ha, 'xy');

reference_masked=label(mask==1);
sum(reference_masked==0)

sum(reference_masked==1)
imshow(mask,[])
if strcmp(application,'wildfire')
    mask_evaluate2(label,mask)
end
imwrite(uint8(mask),strcat('TrainTestMask_',dataset,'.png'));
%%
%mask_evaluate(im,label,mask)


function mask_evaluate2( label, mask )
    %im_size=int64(size(im));        
    %im= reshape(im, im_size(1)*im_size(2), []);    
    mask=reshape(mask, 1, [])';
    label=reshape(label, 1, [])';
    
    label_train=label(mask==1);
    label_test=label(mask==2);
    label_val=label(mask==3);
    
    [values,count]=label_stats_print(label,'train');

    [values,count_train]=label_stats_print(label_train,'train');
    fprintf('Percentage from entire image:%f, %f\n',count_train(1)/count(1),count_train(2)/count(2))
    [values,count_test]=label_stats_print(label_test,'test');
    fprintf('Percentage from entire image:%f, %f\n',count_test(1)/count(1),count_test(2)/count(2))
    [values,count_val]=label_stats_print(label_val,'val');
    fprintf('Percentage from train set:%f, %f\n',count_val(1)/count_train(1),count_val(2)/count_train(2))

    %label_stats_print(label_train,'train');
    
    
end

function [values,count]=label_stats_print(label,subset)
    %[values,count]=unique(label);
    [values,b,c]=unique(uint8(label(:)));
    count = accumarray(c,1);
    ratio=count(2)/count(1);
    fprintf(subset);
    fprintf(': unique label %d:%d %d:%d. Ratio:%d\n',values(1),count(1),values(2),count(2),ratio);
    
end

function mask_evaluate( im, label, mask )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    im_size=int64(size(im));
    for i=1:6
               
        for k=1:6
            tmp=im(:,:,k);
            tmp(im(:,:,i)>32760)=-100;
            tmp(im(:,:,i)<0)=mean(tmp(im(:,:,i)<0));
            im(:,:,k)=tmp;
        end
    end
            
    im= reshape(im, im_size(1)*im_size(2), []);
    mask=reshape(mask, 1, [])';
    label=reshape(label, 1, [])';
    fprintf('im\n');
    stats_print(im(:,:,1));
    im_train=zeros(length(mask(mask==1)),im_size(3));
    im_val=zeros(length(mask(mask==3)),im_size(3));
    im_test=zeros(length(mask(mask==2)),im_size(3));
    for i=1:6
        tmp=im(:,i);
        
        im_train(:,i)=tmp(mask==1);
        im_val(:,i)=tmp(mask==3);
        im_test(:,i)=tmp(mask==2);
    end
  
    fprintf('im_train\n');
    stats_print(im_train(:,:,1));
    fprintf('im_val\n');
    stats_print(im_val(:,:,1));
    fprintf('im_test\n');
    stats_print(im_test(:,:,1));

end

function stats_print(im)
    fprintf('Max=%d,Min=%d,Avg=%d,Std=%d\n',max(im(:)),min(im(:)),mean(im(:)),std(double(im(:))));
end
close all

Nbins = 100;
Nbins_adj = Nbins * 2^(16); % Since only 12 bits are used.

figure(1)
imhist(acre(:,:,2),Nbins_adj);
figure(2)
imhist(para(:,:,2),Nbins_adj);

J=zeros(size(acre));
for i=1:6
    
    J(:,:,i) = imhistmatch(acre(:,:,i),para(:,:,i));
    
end

disp(mean(acre(:)))
disp(mean(para(:)))
disp(mean(J(:)))

save('acre_matched.mat','J');

h5create('dataset.h5','/dataset',size(J));
h5write('dataset.h5','/dataset',J);